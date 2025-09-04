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
API_BASE_URL = os.getenv("API_URL", "https://info.dev.3ceonline.com/ccce/apis")
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

# Exchange rates for currency conversion
EXCHANGE_RATES = {
    'USD': 1.00, 'EUR': 0.85, 'CNY': 7.25, 'JPY': 149.50, 'GBP': 0.73,
    'CAD': 1.37, 'AUD': 1.52, 'KRW': 1320.00, 'INR': 83.20, 'MXN': 17.25,
    'BRL': 5.05, 'VND': 24500.00, 'THB': 35.80, 'MYR': 4.65, 'SGD': 1.35,
    'PHP': 56.20, 'IDR': 15750.00, 'TWD': 31.50, 'HKD': 7.82, 'CHF': 0.88,
    'SEK': 10.85, 'NOK': 10.75, 'DKK': 6.85, 'PLN': 4.05
}


def convert_currency(amount, from_currency, to_currency='USD'):
    """Convert amount from one currency to another using exchange rates"""
    if from_currency == to_currency:
        return amount

    from_rate = EXCHANGE_RATES.get(from_currency, 1.0)
    to_rate = EXCHANGE_RATES.get(to_currency, 1.0)

    # Convert to USD first, then to target currency
    usd_amount = amount / from_rate
    return usd_amount * to_rate


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        request_id = str(uuid.uuid4())
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
    verify_description = data.get("verify_description", False)
    debug_info = f"Request data: {data}\n"

    if not description or not coo:
        debug_info += "Validation failed: Description or COO missing.\n"
        return jsonify({"error": "Description and COO are required", "debug": debug_info}), 400

    try:
        # Step 1: Call 3CEOnline classification API to get HS6 code
        classify_url = f"{API_BASE_URL}/classify/v1/interactive/classify-start"
        classify_headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }
        classify_payload = {
            "proddesc": description
        }

        logger.debug(f"Sending 3CEOnline classification request to {classify_url} with payload: {classify_payload}")
        classify_response = requests.post(classify_url, headers=classify_headers, json=classify_payload, timeout=10)

        logger.debug(f"Classification response status: {classify_response.status_code}")
        logger.debug(f"Classification response text: {classify_response.text}")

        if classify_response.status_code != 200:
            debug_info += f"3CEOnline classification API error ({classify_response.status_code}): {classify_response.text}\n"
            return jsonify({"error": f"Classification API error: {classify_response.text}", "debug": debug_info}), 500

        classify_json = classify_response.json()
        debug_info += f"3CEOnline Classification API Response: {classify_json}\n"

        # Extract HS6 code from classification response
        hs6_code = None
        requires_interaction = False

        if classify_json.get('data'):
            data = classify_json['data']
            hs6_code_raw = data.get('hsCode', '')

            # Check if there's a current question that needs answering
            current_question = data.get('currentQuestionInteraction')
            if current_question and not hs6_code_raw:
                requires_interaction = True
                debug_info += f"3CEOnline requires additional classification questions. Current question: {current_question.get('name', 'unknown')}\n"

            # Only use the HS code if it's not empty
            if hs6_code_raw and hs6_code_raw.strip():
                hs6_code = str(hs6_code_raw)
                logger.debug(f"Successfully extracted HS6 code from classification: {hs6_code}")

        if not hs6_code:
            debug_info += f"No HS6 code found in 3CEOnline classification response. Raw hsCode value: '{classify_json.get('data', {}).get('hsCode', 'N/A')}'\n"

            # Check if this is because interactive classification is needed
            if requires_interaction:
                debug_info += "Classification requires answering additional questions in 3CEOnline interactive system.\n"

            # If verification is enabled and no HS code found, return verification failure
            if verify_description:
                logger.debug(f"Verification enabled and no HS6 code found for description: '{description}'")
                return jsonify({
                    "verification_failed": True,
                    "error": "Description insufficient for classification - may require more specific details or interactive classification",
                    "debug": debug_info
                }), 200
            else:
                # Verification disabled: proceed to Avalara API with description only (no HS6 code)
                logger.debug(
                    f"No HS6 code from 3CEOnline, but verification disabled. Proceeding to Avalara with description only.")
                debug_info += "Proceeding to Avalara quoting API with description only (no HS6 code from classification).\n"

        # Step 2: Call Avalara API (either with HS6 code from classification, or with description only)
        avalara_headers = {
            "Authorization": "Basic " + base64.b64encode(f"{AVALARA_USERNAME}:{AVALARA_PASSWORD}".encode()).decode(),
            "Content-Type": "application/json"
        }

        # Build classification parameters - include HS6 code only if we have one
        classification_params = [{"name": "price", "value": "100", "unit": "USD"}]
        if hs6_code:
            classification_params.append({"name": "hs_code", "value": hs6_code})
            debug_info += f"Adding HS6 code {hs6_code} to Avalara request.\n"
        else:
            debug_info += "No HS6 code available - Avalara will classify based on description only.\n"

        avalara_payload = {
            "id": "classification-request",
            "companyId": int(COMPANY_ID),
            "currency": "USD",
            "sellerCode": "SC8104341",
            "shipFrom": {"country": coo},
            "destinations": [{"shipTo": {"country": "US", "region": "MA"}}],
            "lines": [{
                "lineNumber": 1,
                "quantity": 1,
                "item": {
                    "itemCode": "1",
                    "description": description,
                    "itemGroup": "General",
                    "classificationParameters": classification_params,
                    "parameters": []
                },
                "classificationParameters": classification_params
            }],
            "type": "QUOTE_ENHANCED10",
            "disableCalculationSummary": False,
            "restrictionsCheck": True,
            "program": "Regular"
        }

        logger.debug(f"Sending Avalara request to {AVALARA_API_URL} with payload: {avalara_payload}")
        avalara_response = requests.post(AVALARA_API_URL, headers=avalara_headers, json=avalara_payload, timeout=10)
        avalara_response.raise_for_status()
        avalara_json = avalara_response.json()
        debug_info += f"Avalara API Response: {avalara_json}\n"

        # Extract HS code from Avalara response
        hs_code = avalara_json.get('globalCompliance', [{}])[0].get('quote', {}).get('lines', [{}])[0].get('hsCode')

        if hs_code:
            logger.debug(f"Successfully extracted final HS code from Avalara: {hs_code}")
            return jsonify({"hs_code": hs_code, "debug": debug_info})
        elif hs6_code:
            # Fallback to classified HS6 code if Avalara doesn't provide one
            logger.debug(f"No HS code from Avalara, using classified HS6: {hs6_code}")
            return jsonify({"hs_code": hs6_code, "debug": debug_info})
        else:
            # Neither 3CEOnline nor Avalara provided an HS code
            debug_info += "Neither 3CEOnline classification nor Avalara quoting provided an HS code.\n"
            return jsonify(
                {"error": "No HS code found from either classification or quoting API", "debug": debug_info}), 500

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
    debug_info = f"fetch_stackable_codes called with hs_code: {hs_code}, origin: {origin}, destination: {destination}\n"

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
        debug_info += "Invalid country code format.\n"
        return {"success": False, "error": "Origin and Destination must be 2-letter ISO country codes",
                "debug": debug_info}, 400

    # Truncate HS code to first 6 digits for the API call
    hs_code_for_api = hs_code[:6] if len(hs_code) >= 6 else hs_code

    api_url = f"{API_BASE_URL}/tradedata/import/v1/schedule/{hs_code_for_api}/{origin}/{destination}"
    logger.debug(f"[{request_id}] Calling stackable codes API with truncated HS code: {api_url} (original: {hs_code})")

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            logger.debug(f"[{request_id}] GET request successful")
            response_data = response.json()
            debug_info += f"3CEOnline API Response for stackable codes: {response_data}\n"
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

                # First, extract General rate for fallback and conditional logic
                if 'General' in duties:
                    general_rate_str = duties['General'].get('rate', '')
                    general_percentages = re.findall(r'(\d+(?:\.\d+)?)%', general_rate_str)
                    if general_percentages:
                        general_rate = float(general_percentages[0])
                        logger.debug(f"[{request_id}] Found General rate: {general_rate}%")

                # Handle conditional General rate duties first
                conditional_duty_applied = False
                conditional_rate = 0.0
                conditional_duty_info = None

                # Look for conditional duties and determine which one applies
                for duty_name, duty_info in duties.items():
                    rate_str = duty_info.get('rate', '')

                    if 'if General' in rate_str or 'Apply General' in rate_str:
                        logger.debug(f"[{request_id}] Found conditional General rate logic: {rate_str}")

                        # Parse conditional logic for rates like "15% if General (rate) <15%"
                        if_general_less_match = re.search(r'(\d+(?:\.\d+)?)%\s+if\s+General.*?<\s*(\d+(?:\.\d+)?)%',
                                                          rate_str)
                        # Parse conditional logic for rates like "Apply General (rate) if General (rate) ≥15%"
                        apply_general_match = re.search(r'Apply General.*?≥\s*(\d+(?:\.\d+)?)%', rate_str)

                        if if_general_less_match:
                            fixed_rate = float(if_general_less_match.group(1))
                            threshold = float(if_general_less_match.group(2))

                            if general_rate < threshold:
                                conditional_rate = fixed_rate
                                conditional_duty_info = {
                                    'name': duty_name,
                                    'info': duty_info,
                                    'desc': f"{duty_info.get('longName', duty_name)} (Applied {fixed_rate}% because General rate {general_rate}% < {threshold}%)"
                                }
                                conditional_duty_applied = True
                                logger.debug(
                                    f"[{request_id}] Conditional logic: General rate {general_rate}% < {threshold}%, applying {fixed_rate}%")
                                break  # Stop processing other conditional duties

                        elif apply_general_match:
                            threshold = float(apply_general_match.group(1))

                            if general_rate >= threshold:
                                conditional_rate = general_rate
                                conditional_duty_info = {
                                    'name': duty_name,
                                    'info': duty_info,
                                    'desc': f"{duty_info.get('longName', duty_name)} (Applied General rate {general_rate}% because it is ≥ {threshold}%)"
                                }
                                conditional_duty_applied = True
                                logger.debug(
                                    f"[{request_id}] Conditional logic: General rate {general_rate}% ≥ {threshold}%, applying General rate")
                                break  # Stop processing other conditional duties

                # Add the applicable conditional duty to parsed_duties
                if conditional_duty_applied:
                    duty_name = conditional_duty_info['name']
                    duty_info = conditional_duty_info['info']
                    desc = conditional_duty_info['desc']

                    # Extract Chapter 99 code
                    chapter_99_match = re.search(r'9903\.\d{2}\.\d{2}(?:/\d+)?', duty_name)
                    code_to_use = chapter_99_match.group(0) if chapter_99_match else duty_name

                    parsed_duties[duty_name] = {
                        'code': code_to_use,
                        'desc': desc,
                        'dutyRate': f"{conditional_rate}%"
                    }
                    total_punitive_rate += conditional_rate

                    logger.debug(f"[{request_id}] Applied conditional duty: {duty_name}, rate: {conditional_rate}%")

                # Process remaining non-conditional duties
                for duty_name, duty_info in duties.items():
                    logger.debug(f"[{request_id}] Processing duty: {duty_name}")

                    rate_str = duty_info.get('rate', '')
                    logger.debug(f"[{request_id}] Raw rate string: '{rate_str}'")

                    if not rate_str:
                        logger.debug(f"[{request_id}] No rate found for duty: {duty_name}")
                        continue

                    # Skip conditional duties (already processed) and General duty
                    if 'if General' in rate_str or 'Apply General' in rate_str or duty_name.lower() == 'general':
                        logger.debug(f"[{request_id}] Skipping conditional or general duty: {duty_name}")
                        continue

                    # Standard processing for non-conditional rates
                    # Extract all percentage values from the rate string
                    percentages = re.findall(r'(\d+(?:\.\d+)?)%', rate_str)
                    logger.debug(f"[{request_id}] Extracted percentages from '{rate_str}': {percentages}")

                    if not percentages:
                        logger.debug(f"[{request_id}] No percentages found in rate: {rate_str}")
                        continue

                    # Sum all percentage values for this duty
                    duty_rate = sum(float(p) for p in percentages)
                    desc = duty_info.get('longName', duty_name)

                    logger.debug(f"[{request_id}] Calculated total duty rate: {duty_rate}% for {duty_name}")

                    if duty_rate > 0:
                        # Extract Chapter 99 code
                        chapter_99_match = re.search(r'9903\.\d{2}\.\d{2}(?:/\d+)?', duty_name)
                        code_to_use = chapter_99_match.group(0) if chapter_99_match else duty_name

                        logger.debug(
                            f"[{request_id}] Regex search on '{duty_name}': match={chapter_99_match}, extracted code='{code_to_use}'")

                        parsed_duties[duty_name] = {
                            'code': code_to_use,
                            'desc': desc,
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
                "stackableCodeSets": all_stackable_codes,
                "debug": debug_info
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
            "stackableCodeSets": all_stackable_codes,
            "debug": debug_info
        }, 200

    except requests.RequestException as e:
        debug_info += f"Network error: {str(e)}\n"
        logger.error(f"[{request_id}] Network error: {str(e)}")
        return {"success": False, "error": f"Network error: {str(e)}", "debug": debug_info}, 500
    except Exception as e:
        debug_info += f"Unexpected error: {str(e)}\n"
        logger.error(f"[{request_id}] Unexpected error: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}", "debug": debug_info}, 500


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
        total_value = float(data.get("total_value") or 0)
        calculation_currency = data.get("calculation_currency", "USD")
        debug_info = ""

        logger.debug(f"Received calculation request with {len(products)} products in currency {calculation_currency}")
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
        duty_rate_breakdowns = []  # ADD THIS LINE
        total_duty = 0.0
        logic_applied = ""
        all_rates = []  # Track all rates for max calculation

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

            # Extract debug info from stackable_result if present
            if stackable_result.get("debug"):
                debug_info += f"Stackable codes debug for {hs_code} from {coo}:\n{stackable_result['debug']}\n"
            else:
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

            # Track all rates for finding maximum
            all_rates.append(total_rate)

            stackable_hss = [item.get("code") for item in stackable_codes]
            stackable_hss_desc = [item.get("desc") for item in stackable_codes]

            logger.debug(f"Extracted rates: {rates}, Total rate: {total_rate}%, Stackable codes: {stackable_hss}")

            # Calculate duty for this product line (in USD)
            duty = line_value * (total_rate / 100) if total_rate > 0 else 0

            results.append({
                "description": product.get("description", ""),
                "hs_code": hs_code,
                "coo": coo,
                "currency": calculation_currency,
                "quantity": product.get("quantity", 1),
                "per_unit_value": product.get("per_unit_value", 0),
                "line_value": line_value,
                "original_per_unit_value": product.get("original_per_unit_value", 0),
                "original_line_value": product.get("original_line_value", 0),
                "stackable_hss": stackable_hss,
                "stackable_hss_desc": stackable_hss_desc,
                "rates": rates,
                "total_rate": total_rate,
                "duty": duty
            })

            # ADD THIS SECTION: Build duty rate breakdown for frontend
            if stackable_codes:
                breakdown_stackable_codes = []
                for i, code_item in enumerate(stackable_codes):
                    rate_str = code_item.get("dutyRate", "0%")
                    rate_value = rate_str.strip('%') if rate_str.endswith('%') else rate_str

                    breakdown_stackable_codes.append({
                        "hs_code": code_item.get("code", ""),
                        "description": code_item.get("desc", ""),
                        "rate": rate_value
                    })

                duty_rate_breakdowns.append({
                    "product_description": product.get("description", ""),
                    "hs_code": hs_code,
                    "country_of_origin": coo,
                    "stackable_codes": breakdown_stackable_codes,
                    "total_rate": str(total_rate),
                    "line_value": float(line_value),
                    "duty_amount": float(duty)
                })

        # Calculate total duty based on method (still in USD for internal calculations)
        alternative_savings = None

        if method == "ad_valorem":
            # Use the maximum rate across ALL products
            max_rate = max(all_rates) if all_rates else 0
            total_duty = total_value * (max_rate / 100)
            logic_applied = f"Ad Valorem: Applied max rate {max_rate}% to total shipment value ${total_value}"

            logger.debug(
                f"Ad Valorem calculation: All rates = {all_rates}, Max rate = {max_rate}%, Total value = ${total_value}, Total duty = ${total_duty}")

            # Calculate what specific method would cost
            specific_duty = 0
            if max_rate < 16:
                specific_duty = 80
            elif max_rate <= 25:
                specific_duty = 160
            else:  # max_rate > 25
                specific_duty = 200

            logger.debug(f"Ad valorem duty: ${total_duty:.2f} (${total_value} × {max_rate}%)")
            logger.debug(f"Specific bracket duty: ${specific_duty} (rate {max_rate}% falls in >25% bracket)")

            # Check if specific would save money
            if specific_duty < total_duty:
                savings = total_duty - specific_duty
                alternative_savings = {
                    "method": "Specific Bracket",
                    "alternative_duty": float(specific_duty),
                    "savings": float(savings),
                    "reason": f"Since your maximum rate is {max_rate}%, the specific bracket method would charge a fixed ${specific_duty} instead of {max_rate}% of your ${total_value} shipment value."
                }
                logger.debug(f"Specific bracket would save ${savings:.2f} (${specific_duty} vs ${total_duty:.2f})")
            else:
                logger.debug(f"No savings available: specific bracket ${specific_duty} >= ad valorem ${total_duty:.2f}")

        else:  # specific
            max_rate = max(r["total_rate"] for r in results) if results else 0
            if max_rate < 16:
                total_duty = 80
            elif max_rate <= 25:
                total_duty = 160
            else:
                total_duty = 200
            logic_applied = f"Specific: Max rate {max_rate}% → ${total_duty}"

            # Calculate what ad valorem would cost
            ad_valorem_duty = total_value * (max_rate / 100)

            # Check if ad valorem would save money
            if ad_valorem_duty < total_duty:
                savings = total_duty - ad_valorem_duty
                alternative_savings = {
                    "method": "Ad Valorem",
                    "alternative_duty": float(ad_valorem_duty),
                    "savings": float(savings),
                    "reason": f"Since your shipment value is ${total_value} with a {max_rate}% rate, the ad valorem method would charge {max_rate}% of value (${ad_valorem_duty:.2f}) instead of the fixed ${total_duty} bracket."
                }

        logger.debug(f"Final calculation: Total duty = ${total_duty}")

        response_data = {
            "results": results,
            "total_duty": total_duty,
            "calculation_currency": calculation_currency,
            "logic_applied": logic_applied,
            "duty_rate_breakdowns": duty_rate_breakdowns,  # ADD THIS LINE
            "debug": debug_info
        }

        # Add savings recommendation if applicable
        if alternative_savings:
            response_data["savings_recommendation"] = alternative_savings
            logger.debug(f"Alternative method could save ${alternative_savings['savings']:.2f}")
            logger.debug(f"Adding savings_recommendation to response: {alternative_savings}")
        else:
            logger.debug("No alternative savings available")

        logger.debug(f"Final response keys: {list(response_data.keys())}")
        return jsonify(response_data)

    except Exception as e:
        debug_info += f"Unexpected error: {str(e)}\nStack trace: {traceback.format_exc()}\n"
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "debug": debug_info}), 500


@main.route("/", methods=["GET"])
@auth_required
def home():
    print("Home route hit!")
    return render_template("index.html", today=datetime.now().date().isoformat())