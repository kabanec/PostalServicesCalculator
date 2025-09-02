from flask import Blueprint, render_template, request, jsonify, Response
from functools import wraps
import os
import requests
from datetime import datetime
import re
import uuid
import logging
import traceback
import base64

main = Blueprint("main", __name__)

# API configuration
API_BASE_URL = os.getenv("API_URL", "https://info.dev.3ceonline.com/ccce/apis/tradedata/import/v1/schedule")
API_TOKEN = os.getenv("API_TOKEN", "your_token_here")
AVALARA_USERNAME = os.getenv("AVALARA_USERNAME")
AVALARA_PASSWORD = os.getenv("AVALARA_PASSWORD")
COMPANY_ID = os.getenv("AVALARA_COMPANY_ID")
if COMPANY_ID is None:
    raise ValueError("AVALARA_COMPANY_ID environment variable is not set")
AVALARA_API_URL = f"https://quoting.xbo.dev.avalara.io/api/v2/companies/{COMPANY_ID}/globalcompliance"

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Authentication
VALID_USER = os.getenv("AUTH_USER", "admin")
VALID_PASS = os.getenv("AUTH_PASS", "password")


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        request_id = str(uuid.uuid4())
        # Bypass authentication for HEAD requests (e.g., Render health checks)
        if request.method == 'HEAD':
            logger.debug(f"[{request_id}] Allowing HEAD request without authentication")
            return f(*args, **kwargs)
        auth = request.authorization
        logger.debug(f"[{request_id}] Authorization header: {auth}")
        if not auth:
            logger.error(f"[{request_id}] No authorization header provided")
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        if auth.username != VALID_USER or auth.password != VALID_PASS:
            logger.error(f"[{request_id}] Invalid credentials: username={auth.username}, expected={VALID_USER}")
            return Response('Unauthorized', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        logger.debug(f"[{request_id}] Authentication successful")
        return f(*args, **kwargs)
    return decorated


def is_valid_hts_code(code):
    return bool(re.match(r'^\d{4,10}(\.\d{2})?$|^9903\.\d{2}\.\d{2}$|^98\d{2}\.\d{2}\.\d{2}$', code))


def parse_rate(rate_str):
    """Parse a rate string and return the total punitive percentage, summing all punitive components."""
    if not rate_str or not isinstance(rate_str, str):
        return 0.0
    # Extract all percentage values, focusing on punitive rates
    percentages = re.findall(r'(\d+(?:\.\d+)?)%', rate_str)
    if not percentages:
        return 0.0
    total_rate = sum(float(p.strip('%')) for p in percentages if p)
    # If 'General' is present and it's the last component with multiple rates, exclude it
    if 'general' in rate_str.lower() and len(percentages) > 1:
        total_rate -= float(percentages[-1].strip('%')) if percentages[-1] else 0.0
    return total_rate


@main.route("/fetch_hs_code", methods=["POST"])
@auth_required
def fetch_hs_code():
    data = request.get_json()
    description = data.get("description")
    coo = data.get("coo")
    debug_info = f"Request data: {data}\n"

    if not description or not coo:
        debug_info += "Validation failed: Description or COO missing.\n"
        return jsonify({"error": "Description and COO are required", "debug": debug_info}), 400

    try:
        headers = {
            "Authorization": "Basic " + base64.b64encode(f"{AVALARA_USERNAME}:{AVALARA_PASSWORD}".encode()).decode(),
            "Content-Type": "application/json"
        }
        payload = {
            "id": "classification-request",
            "companyId": int(COMPANY_ID),
            "currency": "USD",
            "sellerCode": "SC8104341",
            "shipFrom": {"country": "GB"},
            "destinations": [{"shipTo": {"country": "US", "region": "MA"}}],
            "lines": [{
                "lineNumber": 1,
                "quantity": 1,
                "item": {
                    "itemCode": "1",
                    "description": description,
                    "itemGroup": "General",
                    "classificationParameters": [{"name": "price", "value": "100", "unit": "USD"}],
                    "parameters": []
                },
                "classificationParameters": [{"name": "price", "value": "100", "unit": "USD"}]
            }],
            "type": "QUOTE_MAXIMUM",
            "disableCalculationSummary": False,
            "restrictionsCheck": True,
            "program": "Regular"
        }

        logger.debug(f"Sending request to {AVALARA_API_URL} with payload: {payload}")
        response = requests.post(AVALARA_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        json_response = response.json()
        debug_info += f"API Response: {json_response}\n"

        hs = json_response.get('globalCompliance', [{}])[0].get('quote', {}).get('lines', [{}])[0].get('hsCode')
        if hs:
            return jsonify({"hs_code": hs})
        else:
            debug_info += "No HS code found in response.\n"
            return jsonify({"error": "No HS code found", "debug": debug_info}), 500

    except requests.RequestException as e:
        debug_info += f"Network error: {str(e)}\nResponse text: {getattr(e.response, 'text', 'No response')}\n"
        logger.error(f"Network error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Network error: {str(e)}", "debug": debug_info}), 500
    except Exception as e:
        debug_info += f"Unexpected error: {str(e)}\nStack trace: {traceback.format_exc()}\n"
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": f"Unexpected error: {str(e)}", "debug": debug_info}), 500


# Embedded fetch_stackable_codes logic
def fetch_stackable_codes(hs_code, origin, destination, request_id):
    if not hs_code or not origin or not destination:
        return {"success": False, "error": "HS Code, Origin, and Destination are required"}, 400

    if not isinstance(hs_code, str) or not isinstance(origin, str) or not isinstance(destination, str):
        return {"success": False, "error": "HS Code, Origin, and Destination must be strings"}, 400

    hs_code = hs_code.strip()
    origin = origin.strip().upper()
    destination = destination.strip().upper()

    if not is_valid_hts_code(hs_code):
        return {"success": False, "error": "Invalid HS Code format"}, 400

    if not re.match(r'^[A-Z]{2}$', origin) or not re.match(r'^[A-Z]{2}$', destination):
        return {"success": False, "error": "Origin and Destination must be 2-letter ISO country codes"}, 400

    # Truncate HS code to first 6 digits for the API call
    hs_code_for_api = hs_code[:6] if len(hs_code) >= 6 else hs_code

    api_url = f"{API_BASE_URL}/{hs_code_for_api}/{origin}/{destination}"
    logger.debug(f"[{request_id}] Calling API with truncated HS code: {api_url} (original: {hs_code})")

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            logger.debug(f"[{request_id}] GET request successful")
            response_data = response.json()
        else:
            logger.error(f"[{request_id}] GET request failed: {response.status_code} - {response.text}")
            return {"success": False, "error": f"API request failed: {response.text}"}, response.status_code

        def find_full_hs_codes_and_duties(data):
            full_hs_codes = []

            def traverse(children, parent_duties=None):
                for item in children or []:
                    code = item.get('code', '')
                    duties = item.get('duties', {})

                    # Process any code that has duties, regardless of length (6-digit, 8-digit, etc.)
                    if duties:
                        logger.debug(f"[{request_id}] Found duties on code: {code} (length: {len(code)})")

                        # If this code has duties, propagate them to child codes that don't have duties
                        def propagate_duties_to_children(children, parent_code, parent_duties):
                            for child in children or []:
                                child_code = child.get('code', '')
                                child_duties = child.get('duties', {})

                                # If child has 6+ digit code but no duties, inherit from parent
                                if child_code and len(child_code) >= 6 and not child_duties:
                                    logger.debug(
                                        f"[{request_id}] Child {child_code} inheriting duties from parent {parent_code}")
                                    process_duties_for_code(child_code, parent_duties)
                                # If child has duties but fewer than parent, merge them
                                elif child_code and len(child_code) >= 6 and child_duties:
                                    logger.debug(
                                        f"[{request_id}] Child {child_code} has its own duties, processing directly")
                                    process_duties_for_code(child_code, child_duties)

                                # Recursively check deeper children
                                propagate_duties_to_children(child.get('children', []), parent_code, parent_duties)

                        # Process duties for this code if it's 6+ digits (covers 6, 8, 10+ digit codes)
                        if code and len(code) >= 6:
                            process_duties_for_code(code, duties)

                        # Propagate duties to children
                        propagate_duties_to_children(item.get('children', []), code, duties)

                    # Continue traversing
                    traverse(item.get('children', []), duties if duties else parent_duties)

            def process_duties_for_code(code, duties):
                # Accept codes that are 6+ digits or valid HTS codes
                if not (len(code) >= 6 or is_valid_hts_code(code)):
                    return

                parsed_duties = {}
                total_punitive_rate = 0.0
                general_rate = 0.0

                logger.debug(f"[{request_id}] Processing HS code: {code} with duties: {list(duties.keys())}")

                # First, extract General rate for fallback
                if 'General' in duties:
                    general_rate_str = duties['General'].get('rate', '')
                    general_percentages = re.findall(r'(\d+(?:\.\d+)?)%', general_rate_str)
                    if general_percentages:
                        general_rate = float(general_percentages[0])
                        logger.debug(f"[{request_id}] Found General rate: {general_rate}%")

                # Process each duty except 'General'
                for duty_name, duty_info in duties.items():
                    logger.debug(f"[{request_id}] Processing duty: {duty_name}")

                    if duty_name.lower() == 'general':
                        logger.debug(f"[{request_id}] Skipping general duty: {duty_name}")
                        continue

                    rate_str = duty_info.get('rate', '')
                    logger.debug(f"[{request_id}] Raw rate string: '{rate_str}'")

                    if not rate_str:
                        logger.debug(f"[{request_id}] No rate found for duty: {duty_name}")
                        continue

                    # Extract all percentage values from the rate string
                    percentages = re.findall(r'(\d+(?:\.\d+)?)%', rate_str)
                    logger.debug(f"[{request_id}] Extracted percentages from '{rate_str}': {percentages}")

                    if not percentages:
                        logger.debug(f"[{request_id}] No percentages found in rate: {rate_str}")
                        continue

                    # Sum all percentage values for this duty
                    duty_rate = sum(float(p) for p in percentages)
                    logger.debug(f"[{request_id}] Calculated total duty rate: {duty_rate}% for {duty_name}")

                    if duty_rate > 0:
                        # Extract Chapter 99 code
                        chapter_99_match = re.search(r'9903\.\d{2}\.\d{2}(?:/\d+)?', duty_name)
                        code_to_use = chapter_99_match.group(0) if chapter_99_match else duty_name

                        logger.debug(
                            f"[{request_id}] Regex search on '{duty_name}': match={chapter_99_match}, extracted code='{code_to_use}'")

                        parsed_duties[duty_name] = {
                            'code': code_to_use,
                            'desc': duty_info.get('longName', duty_name),
                            'dutyRate': f"{duty_rate}%"
                        }
                        total_punitive_rate += duty_rate

                        logger.debug(
                            f"[{request_id}] Successfully parsed duty: {duty_name}, code: {code_to_use}, rate: {duty_rate}%")
                        logger.debug(f"[{request_id}] Added to parsed_duties: {parsed_duties[duty_name]}")

                # Add to results if we found duties (but only for final HS codes, not intermediate ones)
                if parsed_duties and len(code) >= 10:
                    logger.debug(
                        f"[{request_id}] Found {len(parsed_duties)} punitive duties for {code}, total rate: {total_punitive_rate}%")
                    full_hs_codes.append({
                        'code': code,
                        'duties': parsed_duties,
                        'punitiveRate': f"{total_punitive_rate}%"
                    })
                elif general_rate > 0 and len(code) >= 10:
                    logger.debug(
                        f"[{request_id}] No punitive duties found for {code}, using General rate: {general_rate}%")
                    parsed_duties['General'] = {
                        'code': code,
                        'desc': f"General tariff rate for {code}",
                        'dutyRate': f"{general_rate}%"
                    }
                    full_hs_codes.append({
                        'code': code,
                        'duties': parsed_duties,
                        'punitiveRate': f"{general_rate}%"
                    })
                else:
                    logger.debug(f"[{request_id}] Processed duties for {code} (intermediate level)")

            traverse(data.get('children', []))
            return full_hs_codes

        hs_code_duties = find_full_hs_codes_and_duties(response_data)

        # Handle case where no duties are found
        if not hs_code_duties:
            logger.debug(f"[{request_id}] No HS codes with duties found, returning zero-duty result")
            all_stackable_codes = [{
                'primaryHTS': hs_code,
                'stackableCodes': [],
                'punitiveRate': '0%'
            }]

            return {
                "success": True,
                "data": response_data,
                "stackableCodeSets": all_stackable_codes
            }, 200

        all_stackable_codes = []
        for hs_item in hs_code_duties:
            primary_hts = hs_item['code']
            duties = hs_item['duties']
            punitive_rate = hs_item['punitiveRate']

            # Create stackable codes list directly from duties
            chapter_99_tariff_codes = []
            for duty_name, duty_info in duties.items():
                chapter_99_tariff_codes.append({
                    'code': duty_info['code'],
                    'desc': duty_info['desc'],
                    'rate': duty_info['dutyRate']
                })
                logger.debug(
                    f"[{request_id}] Added to chapter_99_tariff_codes: {duty_info['code']} with rate {duty_info['dutyRate']}")

            logger.debug(f"[{request_id}] Final chapter_99_tariff_codes: {chapter_99_tariff_codes}")

            # Sort by priority rules
            sorted_codes = sort_chapter_99_codes(chapter_99_tariff_codes)
            ordered_hts_codes = order_stackable_hts_codes(primary_hts, [], [], sorted_codes)

            all_stackable_codes.append({
                'primaryHTS': primary_hts,
                'stackableCodes': ordered_hts_codes,
                'punitiveRate': punitive_rate
            })

            logger.debug(f"[{request_id}] Final stackableCodes for {primary_hts}: {ordered_hts_codes}")

        return {
            "success": True,
            "data": response_data,
            "stackableCodeSets": all_stackable_codes
        }, 200

    except requests.RequestException as e:
        logger.error(f"[{request_id}] Network error: {str(e)}")
        return {"success": False, "error": f"Network error: {str(e)}"}, 500
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}, 500


def order_stackable_hts_codes(primary_hts, chapter_98_codes, exemption_codes, chapter_99_tariff_codes):
    ordered_hts_codes = []
    for item in chapter_99_tariff_codes:
        # Accept any code - don't filter based on validation
        logger.debug(f"Processing stackable code: {item['code']} with validation check")
        ordered_hts_codes.append({
            'code': item['code'],
            'desc': item['desc'],
            'dutyRate': item['rate']
        })
        logger.debug(f"Added stackable code: {item['code']} with rate {item['rate']}")

    logger.debug(f"Final ordered_hts_codes: {ordered_hts_codes}")
    return ordered_hts_codes


def sort_chapter_99_codes(chapter_99_tariff_codes):
    tariff_priority_rules = {
        '9903.88.01': 1,
        '9903.01.25': 2,
        '9903.88.15': 3,
        '9903.94.05': 4,
        '9903.81.90': 5,
        '9903.88.03/04': 6
    }
    return sorted(chapter_99_tariff_codes, key=lambda x: tariff_priority_rules.get(x['code'], 999))


@main.route("/calculate", methods=["POST"])
@auth_required
def calculate_postal_duty():
    try:
        data = request.get_json()
        entry_date = datetime.strptime(data.get("entry_date"), "%Y-%m-%d")
        method = data.get("method", "ad_valorem")
        products = data.get("products", [])
        total_value = data.get("total_value", 0)
        debug_info = ""

        logger.debug(f"Received calculation request with {len(products)} products")
        for i, product in enumerate(products):
            logger.debug(
                f"Product {i}: {product.get('description')} from {product.get('coo')} with HS {product.get('hs_code')}")

        if entry_date < datetime(2025, 8, 29):
            return jsonify({"error": "Entry date is out of scope for EO."}), 400

        if entry_date >= datetime(2026, 2, 28) and method == "specific":
            method = "ad_valorem"
            debug_info += "Method forced to ad_valorem due to date.\n"

        if total_value > 2500:
            return jsonify({"error": "Total shipment value exceeds $2,500. Postal method not available."}), 400

        results = []
        total_duty = 0.0
        logic_applied = ""
        coo_buckets = {}

        for product in products:
            hs_code = product.get("hs_code")
            coo = product.get("coo")
            line_value = product.get("line_value", 0)

            logger.debug(
                f"Processing product: {product.get('description')} with HS: {hs_code}, COO: {coo}, Line Value: {line_value}")

            if not hs_code or not coo:
                debug_info += f"Validation failed for product {product.get('description')}: HS Code or COO missing.\n"
                return jsonify({"error": "All products must have HS Code and COO.", "debug": debug_info}), 400

            request_id = str(uuid.uuid4())
            stackable_result, status_code = fetch_stackable_codes(hs_code, coo, "US", request_id)
            debug_info += f"Stackable result for {hs_code} from {coo}: {stackable_result}\n"

            if not stackable_result.get("success"):
                return jsonify({"error": stackable_result.get("error"), "debug": debug_info}), status_code

            stackable_codes = stackable_result.get("stackableCodeSets", [{}])[0].get("stackableCodes", [])
            rates = [item.get("dutyRate", "0%") for item in stackable_codes if item.get("dutyRate")]

            # Calculate total rate from all stackable codes
            total_rate = 0.0
            for rate_str in rates:
                if rate_str and rate_str.endswith('%'):
                    rate_value = float(rate_str.strip('%'))
                    total_rate += rate_value

            stackable_hss = [item.get("code") for item in stackable_codes]
            stackable_hss_desc = [item.get("desc") for item in stackable_codes]

            logger.debug(f"Extracted rates: {rates}, Total rate: {total_rate}%, Stackable codes: {stackable_hss}")

            if method == "ad_valorem":
                if coo not in coo_buckets:
                    coo_buckets[coo] = {"value": 0, "rate": total_rate}
                else:
                    # Update rate if higher (for mixed products from same COO)
                    coo_buckets[coo]["rate"] = max(coo_buckets[coo]["rate"], total_rate)
                coo_buckets[coo]["value"] += line_value
                logic_applied = "Ad Valorem: Summed duties by COO buckets."

            # Calculate duty for this product line
            duty = line_value * (total_rate / 100) if total_rate > 0 else 0

            results.append({
                "description": product.get("description", ""),
                "hs_code": hs_code,
                "coo": coo,
                "quantity": product.get("quantity", 1),
                "per_unit_value": product.get("per_unit_value", 0),
                "line_value": line_value,
                "stackable_hss": stackable_hss,
                "stackable_hss_desc": stackable_hss_desc,
                "rates": rates,
                "total_rate": total_rate,
                "duty": duty
            })

        # Calculate total duty based on method
        if method == "ad_valorem":
            total_duty = 0.0
            for bucket_coo, bucket_data in coo_buckets.items():
                bucket_duty = bucket_data["value"] * (bucket_data["rate"] / 100)
                total_duty += bucket_duty
                logger.debug(
                    f"COO bucket {bucket_coo}: Value=${bucket_data['value']}, Rate={bucket_data['rate']}%, Duty=${bucket_duty}")
        else:  # specific
            max_rate = max(r["total_rate"] for r in results) if results else 0
            if max_rate < 16:
                total_duty = 80
            elif max_rate <= 25:
                total_duty = 160
            else:
                total_duty = 200
            logic_applied = f"Specific: Max rate {max_rate}% → ${total_duty}"

        logger.debug(f"Final calculation: Total duty = ${total_duty}")

        return jsonify({
            "results": results,
            "total_duty": total_duty,
            "logic_applied": logic_applied,
            "debug": debug_info
        })

    except Exception as e:
        debug_info += f"Unexpected error: {str(e)}\nStack trace: {traceback.format_exc()}\n"
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "debug": debug_info}), 500


@main.route("/", methods=["GET"])
@auth_required
def home():
    print("Home route hit!")
    return render_template("index.html", today=datetime.now().date().isoformat())