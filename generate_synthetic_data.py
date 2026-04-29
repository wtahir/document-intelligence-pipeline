# generate_synthetic_data.py
# Generates realistic synthetic German insurance claim PDFs
# for water damage, storm damage, and glass damage scenarios.
#
# Each claim case produces 3 PDFs:
#   1. Email — claimant reports the damage
#   2. Invoice — repair cost with line items
#   3. Photo description — damage documentation
#
# Also generates policy_metadata.json — the source of truth for
# coverage checks. Some claimants are covered, some are not.

import os
import json
import random
from datetime import datetime, timedelta
from fpdf import FPDF

os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# ─── Claimant pool (fixed names so policy metadata can reference them) ────

CLAIMANTS = [
    {"first": "Thomas", "last": "Mueller", "city": "Berlin", "zip": "10115"},
    {"first": "Maria", "last": "Schmidt", "city": "Munich", "zip": "80331"},
    {"first": "Klaus", "last": "Weber", "city": "Hamburg", "zip": "20095"},
    {"first": "Andrea", "last": "Wagner", "city": "Frankfurt", "zip": "60311"},
    {"first": "Michael", "last": "Becker", "city": "Cologne", "zip": "50667"},
    {"first": "Sabine", "last": "Hoffmann", "city": "Stuttgart", "zip": "70173"},
    {"first": "Wolfgang", "last": "Fischer", "city": "Dortmund", "zip": "44135"},
    {"first": "Elisabeth", "last": "Schwarz", "city": "Dresden", "zip": "01067"},
    {"first": "Stefan", "last": "Bauer", "city": "Nuremberg", "zip": "90402"},
    {"first": "Monika", "last": "Steiner", "city": "Hanover", "zip": "30159"},
    {"first": "Peter", "last": "Gruber", "city": "Leipzig", "zip": "04109"},
    {"first": "Ingrid", "last": "Huber", "city": "Essen", "zip": "45127"},
    {"first": "Christian", "last": "Maier", "city": "Bremen", "zip": "28195"},
    {"first": "Ursula", "last": "Berger", "city": "Bonn", "zip": "53111"},
    {"first": "Markus", "last": "Leitner", "city": "Mainz", "zip": "55116"},
    {"first": "Anna", "last": "Koch", "city": "Freiburg", "zip": "79098"},
    {"first": "Hans", "last": "Richter", "city": "Augsburg", "zip": "86150"},
    {"first": "Petra", "last": "Wolf", "city": "Wiesbaden", "zip": "65183"},
    {"first": "Georg", "last": "Braun", "city": "Mannheim", "zip": "68159"},
    {"first": "Claudia", "last": "Zimmermann", "city": "Karlsruhe", "zip": "76131"},
]

# ─── Damage scenarios ──────────────────────────────────────────────────────

WATER_DAMAGE_OBJECTS = [
    "kitchen ceiling", "basement flooring", "bathroom walls",
    "living room parquet", "laundry room floor", "hallway ceiling",
    "bedroom carpet", "utility room walls", "dining room floor",
    "garage concrete floor",
]

WATER_DAMAGE_DESCRIPTIONS = [
    "A burst pipe in the bathroom caused extensive flooding. Water seeped through the floor and damaged the {object} below. The {object} shows visible water stains, warping, and mold formation.",
    "The washing machine supply hose ruptured overnight. Approximately 200 liters of water flooded the area, causing severe damage to the {object}. Emergency plumber was called to stop the leak.",
    "A corroded copper pipe in the wall burst during freezing temperatures. The resulting water leak damaged the {object} significantly. Drying equipment has been installed.",
    "Leaking dishwasher connection caused slow water damage over several weeks. The {object} shows extensive water damage including swelling and discoloration. Professional repair is required.",
    "Toilet overflow due to a blocked drain caused water to spread across the floor. The {object} absorbed water and now shows buckling and staining. Immediate replacement is needed.",
]

STORM_DAMAGE_OBJECTS = [
    "roof tiles", "garden shed", "terrace canopy",
    "car port structure", "fence panels", "window shutters",
    "chimney cap", "solar panels", "satellite dish",
    "balcony railing",
]

STORM_DAMAGE_DESCRIPTIONS = [
    "Severe storm with wind speeds exceeding 100 km/h tore off several {object}. Debris was scattered across the property. Temporary covering has been applied to prevent further water ingress.",
    "A large tree branch fell during the storm and crashed into the {object}, causing significant structural damage. The {object} is no longer functional and requires complete replacement.",
    "Storm-force winds ripped the {object} from its mounting. The {object} landed in the neighbor's garden. No injuries reported but the property damage is substantial.",
    "Hailstorm with golf-ball sized hail stones severely damaged the {object}. Multiple impact marks and cracks are visible. A professional assessment confirms total replacement is needed.",
    "Tornado-like wind conditions caused the {object} to collapse entirely. The structural integrity is compromised beyond repair. Emergency securing measures have been taken.",
]

GLASS_DAMAGE_OBJECTS = [
    "front door glass panel", "living room window pane",
    "kitchen window", "shower glass door", "balcony glass door",
    "skylight window", "conservatory glass roof panel",
    "bathroom mirror wall", "shop front window", "patio sliding door glass",
]

GLASS_DAMAGE_DESCRIPTIONS = [
    "The {object} shattered unexpectedly due to thermal stress. Glass fragments were found across the room. Temporary boarding has been applied for security. Professional glazier assessment confirms the {object} needs full replacement.",
    "A stray ball from neighboring children struck the {object}, causing it to crack and partially shatter. The {object} is now a safety hazard and must be replaced immediately.",
    "During a break-in attempt, the intruder smashed the {object}. While nothing was stolen, the {object} is completely destroyed. Police report has been filed (reference included).",
    "Unknown cause led to spontaneous breakage of the {object}. The tempered glass crumbled into small pieces. Building management confirms this is a known issue with this glass type in older installations.",
    "A falling object from the floor above hit the {object}, shattering it completely. The {object} pieces have been safely removed. Urgent replacement is required for weather protection.",
]

REPAIR_VENDORS = {
    "water": [
        "Rohrfix GmbH - Sanitaer und Heizung",
        "AquaService Bauer - Leckortung und Sanierung",
        "Mueller Installationstechnik",
        "WasserProfi Berlin - Rohrsanierung",
        "Schmidt und Sohn Klempnerbetrieb",
    ],
    "storm": [
        "Dachdeckerei Sturm und Partner",
        "Bauservice Wagner - Sturmschadenbeseitigung",
        "HandwerksProfis Nord GmbH",
        "Fischer Bau und Sanierung",
        "Meisterbetrieb Hoffmann - Dach und Fassade",
    ],
    "glass": [
        "Glaserei Kristall GmbH",
        "FensterExperte Weber",
        "Glas Schmidt - Notdienst und Einbau",
        "Bayer Glasservice",
        "ProGlas Fensterbau und Verglasung",
    ],
}

INVOICE_LINE_ITEMS = {
    "water": [
        ("Emergency call-out and leak detection", 180, 350),
        ("Water extraction and drying equipment rental", 400, 900),
        ("Pipe repair / replacement", 250, 800),
        ("Floor / ceiling removal and disposal", 300, 1200),
        ("New flooring / ceiling material and installation", 800, 4500),
        ("Mold treatment and prevention", 200, 600),
        ("Repainting and finishing", 300, 1000),
    ],
    "storm": [
        ("Emergency securing and temporary cover", 200, 500),
        ("Debris removal and disposal", 150, 600),
        ("Structural damage assessment", 250, 450),
        ("Material replacement", 500, 5000),
        ("Installation and labor", 400, 3000),
        ("Scaffolding rental", 300, 900),
        ("Final inspection and quality check", 100, 250),
    ],
    "glass": [
        ("Emergency boarding and securing", 120, 280),
        ("Glass removal and disposal", 80, 200),
        ("New glass panel (tempered/laminated)", 300, 2500),
        ("Frame repair / adjustment", 100, 400),
        ("Professional installation", 200, 600),
        ("Sealant and weatherproofing", 50, 150),
        ("Cleanup and final inspection", 60, 120),
    ],
}


# ─── Helpers ──────────────────────────────────────────────────────────────

def random_date(start_year=2024, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return start + timedelta(days=random.randint(0, (end - start).days))


def generate_claim_number(damage_type):
    prefix = {"water": "WD", "storm": "SD", "glass": "GD"}[damage_type]
    year = random.choice([2024, 2025])
    seq = random.randint(10000, 99999)
    return f"{prefix}-{year}-{seq}"


def generate_policy_number():
    prefix = random.choice(["POL", "HH", "WG"])
    return f"{prefix}-{random.randint(100000, 999999)}"


def generate_email(first, last):
    domain = random.choice(["gmail.com", "outlook.com", "gmx.de", "web.de"])
    return f"{first.lower()}.{last.lower()}@{domain}"


def generate_phone():
    return f"+49 {random.randint(151, 179)} {random.randint(1000000, 9999999)}"


def generate_iban():
    return f"DE{random.randint(10, 99)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(10, 99)}"


# ─── PDF creation ─────────────────────────────────────────────────────────

def create_pdf(content, output_path):
    """Creates a single-page PDF with the given text content."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    safe = content.encode('latin-1', errors='replace').decode('latin-1')
    width = pdf.w - pdf.l_margin - pdf.r_margin
    for line in safe.split('\n'):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(width, 5, line)
    pdf.output(output_path)


# ─── Document generators ─────────────────────────────────────────────────

def generate_email_pdf(claim_number, claimant, damage_type, damaged_object,
                       damage_desc, damage_date, policy_number):
    """Generate a claim notification email PDF."""
    full_name = f"{claimant['first']} {claimant['last']}"
    email = generate_email(claimant['first'], claimant['last'])
    phone = generate_phone()
    report_date = damage_date + timedelta(days=random.randint(1, 5))

    damage_type_label = {
        "water": "Water Damage (Leitungswasserschaden)",
        "storm": "Storm Damage (Sturmschaden)",
        "glass": "Glass Damage (Glasbruch)",
    }[damage_type]

    content = f"""INSURANCE CLAIM NOTIFICATION

Date: {report_date.strftime('%Y-%m-%d')}
From: {email}
To: claims@versicherung-deutschland.de
Subject: Claim Notification - {claim_number}

Dear Claims Department,

I am writing to report a damage incident and submit a claim under my insurance policy.

CLAIMANT DETAILS:
Name: {full_name}
Address: {claimant['zip']} {claimant['city']}
Phone: {phone}
Email: {email}
Policy Number: {policy_number}

CLAIM DETAILS:
Claim Number: {claim_number}
Damage Type: {damage_type_label}
Damaged Object: {damaged_object}
Date of Damage: {damage_date.strftime('%Y-%m-%d')}

DAMAGE DESCRIPTION:
{damage_desc}

I have attached photographs of the damage and will forward the repair invoice once the work is completed.

I kindly request your prompt assessment and processing of this claim.

Best regards,
{full_name}
{phone}
{email}
"""
    return content, report_date


def generate_invoice_pdf(claim_number, claimant, damage_type, damaged_object,
                         damage_date, policy_number):
    """Generate a repair invoice PDF with line items and total."""
    full_name = f"{claimant['first']} {claimant['last']}"
    vendor = random.choice(REPAIR_VENDORS[damage_type])
    invoice_date = damage_date + timedelta(days=random.randint(10, 45))
    invoice_number = f"INV-{random.randint(10000, 99999)}"

    # Generate 3-5 random line items
    available_items = INVOICE_LINE_ITEMS[damage_type]
    num_items = random.randint(3, min(5, len(available_items)))
    selected_items = random.sample(available_items, num_items)

    lines = []
    total = 0
    for desc, low, high in selected_items:
        amount = round(random.uniform(low, high), 2)
        total += amount
        lines.append((desc, amount))

    tax = round(total * 0.19, 2)
    grand_total = round(total + tax, 2)

    items_text = ""
    for i, (desc, amount) in enumerate(lines, 1):
        items_text += f"  {i}. {desc:.<55} EUR {amount:>10,.2f}\n"

    content = f"""REPAIR INVOICE

{vendor}
Invoice Date: {invoice_date.strftime('%Y-%m-%d')}
Invoice Number: {invoice_number}

BILL TO:
{full_name}
{claimant['zip']} {claimant['city']}

REFERENCE:
Claim Number: {claim_number}
Policy Number: {policy_number}
Damage Type: {damage_type.title()} Damage
Damaged Object: {damaged_object}
Date of Damage: {damage_date.strftime('%Y-%m-%d')}

SERVICES RENDERED:
{items_text}
  Subtotal:                                                EUR {total:>10,.2f}
  VAT (19%):                                               EUR {tax:>10,.2f}
  TOTAL AMOUNT:                                            EUR {grand_total:>10,.2f}

PAYMENT DETAILS:
Payable within 30 days to:
IBAN: {generate_iban()}
Reference: {claim_number}

This invoice covers all labor, materials, and disposal costs for the repair
of the {damaged_object} damaged on {damage_date.strftime('%Y-%m-%d')}.

{vendor}
"""
    return content, grand_total, invoice_date


def generate_photo_pdf(claim_number, claimant, damage_type, damaged_object,
                       damage_desc, damage_date):
    """Generate a photo documentation PDF (describes what photos show)."""
    full_name = f"{claimant['first']} {claimant['last']}"
    photo_date = damage_date + timedelta(days=random.randint(0, 3))

    photo_descriptions = {
        "water": [
            f"Photo 1: Overview of the damaged {damaged_object} showing water stain patterns and discoloration across the entire surface.",
            f"Photo 2: Close-up of the burst/leaking pipe connection point. Visible corrosion and water residue around the joint.",
            f"Photo 3: Detail shot of water damage to the {damaged_object} showing material swelling and deformation.",
            f"Photo 4: Wide angle view showing the extent of flooding and water marks on surrounding walls.",
            f"Photo 5: Moisture meter reading on the {damaged_object} surface showing elevated humidity levels.",
        ],
        "storm": [
            f"Photo 1: Overview of storm damage to the {damaged_object}. Visible structural displacement and debris.",
            f"Photo 2: Close-up of the impact point where wind/debris struck the {damaged_object}.",
            f"Photo 3: Fallen tree branch / debris that caused the damage to the {damaged_object}.",
            f"Photo 4: Wide angle of the property showing overall storm damage context.",
            f"Photo 5: Detail of the {damaged_object} showing cracks, breaks, and structural compromise.",
        ],
        "glass": [
            f"Photo 1: Full view of the shattered {damaged_object}. Spider-web crack pattern visible across the entire pane.",
            f"Photo 2: Close-up of the point of impact on the {damaged_object}. Clear radial fracture lines emanating from center.",
            f"Photo 3: Glass fragments on the floor below the {damaged_object}. Safety concern documented.",
            f"Photo 4: The frame and mounting of the {damaged_object} showing condition for replacement assessment.",
            f"Photo 5: Temporary boarding/covering installed over the damaged {damaged_object} for security.",
        ],
    }

    photos = photo_descriptions[damage_type]
    photos_text = "\n\n".join(photos)

    content = f"""DAMAGE PHOTO DOCUMENTATION

Claim Number: {claim_number}
Claimant: {full_name}
Date of Photos: {photo_date.strftime('%Y-%m-%d')}
Damage Type: {damage_type.title()} Damage
Damaged Object: {damaged_object}
Location: {claimant['zip']} {claimant['city']}

PHOTO EVIDENCE:

{photos_text}

DAMAGE ASSESSMENT NOTES:
The photographs above document the damage to the {damaged_object} at the
property of {full_name} in {claimant['city']}.

{damage_desc}

The damage was first discovered on {damage_date.strftime('%Y-%m-%d')}.
These photographs were taken on {photo_date.strftime('%Y-%m-%d')} to document
the current condition for insurance claim processing.

Assessment: The {damaged_object} requires {'complete replacement' if random.random() > 0.3 else 'professional repair'}.

Documented by: {full_name}
"""
    return content, photo_date


# ─── Policy metadata generator ───────────────────────────────────────────

def generate_policy_metadata():
    """
    Generate policy_metadata.json -- the source of truth for coverage checks.

    Not every claimant is covered for every damage type. This is intentional:
    it lets the pipeline test whether a claim should be approved or denied.

    Coverage types: water_damage, storm_damage, glass_damage
    Some policies cover all three, some only one or two.
    """
    policies = []

    coverage_combinations = [
        ["water_damage", "storm_damage", "glass_damage"],  # full coverage
        ["water_damage", "storm_damage"],                    # no glass
        ["water_damage", "glass_damage"],                    # no storm
        ["storm_damage", "glass_damage"],                    # no water
        ["water_damage"],                                     # water only
        ["storm_damage"],                                     # storm only
        ["glass_damage"],                                     # glass only
    ]

    # Items covered per damage type — not all items are covered by every policy.
    # This creates realistic denial scenarios where the damage TYPE is covered
    # but the specific OBJECT is excluded.
    COVERED_ITEMS = {
        "water_damage": [
            ["kitchen ceiling", "basement flooring", "bathroom walls", "living room parquet",
             "laundry room floor", "hallway ceiling", "bedroom carpet", "utility room walls",
             "dining room floor", "garage concrete floor"],                        # full
            ["kitchen ceiling", "basement flooring", "bathroom walls",
             "living room parquet", "laundry room floor"],                          # partial
            ["kitchen ceiling", "bathroom walls", "hallway ceiling"],               # minimal
        ],
        "storm_damage": [
            ["roof tiles", "garden shed", "terrace canopy", "car port structure",
             "fence panels", "window shutters", "chimney cap", "solar panels",
             "satellite dish", "balcony railing"],                                 # full
            ["roof tiles", "garden shed", "terrace canopy",
             "fence panels", "window shutters"],                                   # partial
            ["roof tiles", "chimney cap", "satellite dish"],                       # minimal
        ],
        "glass_damage": [
            ["front door glass panel", "living room window pane", "kitchen window",
             "shower glass door", "balcony glass door", "skylight window",
             "conservatory glass roof panel", "bathroom mirror wall",
             "shop front window", "patio sliding door glass"],                     # full
            ["front door glass panel", "living room window pane",
             "kitchen window", "shower glass door", "balcony glass door"],          # partial
            ["front door glass panel", "living room window pane",
             "kitchen window"],                                                     # minimal
        ],
    }

    for i, claimant in enumerate(CLAIMANTS):
        full_name = f"{claimant['first']} {claimant['last']}"
        policy_number = f"POL-{100000 + i * 1111}"

        # Weight toward full coverage but ensure some gaps
        if i < 8:
            coverage = coverage_combinations[0]  # full coverage
        elif i < 12:
            coverage = coverage_combinations[i - 8 + 1]  # partial
        elif i < 16:
            coverage = random.choice(coverage_combinations[1:4])  # partial
        else:
            coverage = random.choice(coverage_combinations[4:])  # minimal

        start_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365))
        end_date = start_date + timedelta(days=365 * 2)

        # Build covered_items based on coverage types
        covered_items = {}
        for ctype in coverage:
            # First 8 claimants get full item lists, mid-range get partial, rest minimal
            if i < 8:
                item_list = COVERED_ITEMS[ctype][0]  # full
            elif i < 14:
                item_list = COVERED_ITEMS[ctype][1]  # partial
            else:
                item_list = COVERED_ITEMS[ctype][2]  # minimal
            covered_items[ctype] = item_list

        policy = {
            "policy_number": policy_number,
            "policyholder_name": full_name,
            "address": f"{claimant['zip']} {claimant['city']}",
            "email": generate_email(claimant['first'], claimant['last']),
            "coverage_types": coverage,
            "covered_items": covered_items,
            "coverage_limit_eur": random.choice([25000, 50000, 75000, 100000, 150000]),
            "deductible_eur": random.choice([250, 500, 750, 1000]),
            "policy_start": start_date.strftime('%Y-%m-%d'),
            "policy_end": end_date.strftime('%Y-%m-%d'),
            "status": "active",
            "premium_monthly_eur": round(random.uniform(35, 120), 2),
        }
        policies.append(policy)

    return policies


# ─── Main generator ──────────────────────────────────────────────────────

def generate_dataset(claims_per_type=10):
    """
    Generates the full synthetic dataset:
    - 3 PDFs per claim (email + invoice + photo) x 3 damage types x claims_per_type
    - policy_metadata.json for coverage checks
    """
    total_claims = claims_per_type * 3
    total_pdfs = total_claims * 3

    print(f"Generating {total_claims} claim cases ({claims_per_type} per damage type)")
    print(f"Total PDFs to create: {total_pdfs}")
    print(f"Damage types: Water, Storm, Glass")
    print()

    # Clear existing PDFs
    pdf_dir = "data/pdfs"
    for f in os.listdir(pdf_dir) if os.path.exists(pdf_dir) else []:
        if f.endswith('.pdf'):
            os.remove(os.path.join(pdf_dir, f))
    os.makedirs(pdf_dir, exist_ok=True)

    # Generate policy metadata first
    policies = generate_policy_metadata()
    policy_path = "data/policy_metadata.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)
    print(f"Generated policy metadata: {policy_path} ({len(policies)} policies)")
    print()

    damage_types = ["water", "storm", "glass"]
    damage_objects = {
        "water": WATER_DAMAGE_OBJECTS,
        "storm": STORM_DAMAGE_OBJECTS,
        "glass": GLASS_DAMAGE_OBJECTS,
    }
    damage_descs = {
        "water": WATER_DAMAGE_DESCRIPTIONS,
        "storm": STORM_DAMAGE_DESCRIPTIONS,
        "glass": GLASS_DAMAGE_DESCRIPTIONS,
    }

    claim_registry = []
    pdf_count = 0

    for damage_type in damage_types:
        print(f"Generating {claims_per_type} {damage_type} damage claims...")

        for i in range(claims_per_type):
            claimant = random.choice(CLAIMANTS)
            full_name = f"{claimant['first']} {claimant['last']}"
            claim_number = generate_claim_number(damage_type)
            damaged_object = random.choice(damage_objects[damage_type])
            damage_date = random_date()

            # Find this claimant's policy
            policy = next(
                (p for p in policies if p["policyholder_name"] == full_name),
                None
            )
            policy_number = policy["policy_number"] if policy else "UNKNOWN"

            # Description with object inserted
            desc_template = random.choice(damage_descs[damage_type])
            damage_desc = desc_template.format(object=damaged_object)

            # 1. Email PDF
            email_content, email_date = generate_email_pdf(
                claim_number, claimant, damage_type, damaged_object,
                damage_desc, damage_date, policy_number
            )
            email_filename = f"{claim_number}_email.pdf"
            create_pdf(email_content, os.path.join(pdf_dir, email_filename))
            pdf_count += 1

            # 2. Invoice PDF
            invoice_content, invoice_total, invoice_date = generate_invoice_pdf(
                claim_number, claimant, damage_type, damaged_object,
                damage_date, policy_number
            )
            invoice_filename = f"{claim_number}_invoice.pdf"
            create_pdf(invoice_content, os.path.join(pdf_dir, invoice_filename))
            pdf_count += 1

            # 3. Photo documentation PDF
            photo_content, photo_date = generate_photo_pdf(
                claim_number, claimant, damage_type, damaged_object,
                damage_desc, damage_date
            )
            photo_filename = f"{claim_number}_photo.pdf"
            create_pdf(photo_content, os.path.join(pdf_dir, photo_filename))
            pdf_count += 1

            # Track claim info
            damage_type_coverage_key = f"{damage_type}_damage"
            is_covered = (
                policy is not None
                and damage_type_coverage_key in policy.get("coverage_types", [])
            )

            claim_registry.append({
                "claim_number": claim_number,
                "claimant_name": full_name,
                "damage_type": damage_type,
                "damaged_object": damaged_object,
                "damage_date": damage_date.strftime('%Y-%m-%d'),
                "invoice_amount_eur": invoice_total,
                "policy_number": policy_number,
                "is_covered": is_covered,
                "files": [email_filename, invoice_filename, photo_filename],
            })

        print(f"  {claims_per_type} {damage_type} claims done ({claims_per_type * 3} PDFs)")

    # Save claim registry (ground truth for evaluation)
    registry_path = "data/output/claim_registry.json"
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(claim_registry, f, indent=2, ensure_ascii=False)

    print(f"\nDONE: {pdf_count} PDFs generated in {pdf_dir}/")
    print(f"Claim registry: {registry_path} ({len(claim_registry)} claims)")
    print(f"Policy metadata: {policy_path} ({len(policies)} policies)")
    print(f"\nBreakdown:")
    for dt in damage_types:
        count = sum(1 for c in claim_registry if c["damage_type"] == dt)
        covered = sum(1 for c in claim_registry if c["damage_type"] == dt and c["is_covered"])
        print(f"  {dt.title():10s}: {count} claims ({covered} covered, {count - covered} NOT covered)")
    print(f"\nReady for pipeline -- run: python stage1_ingestion.py")


if __name__ == "__main__":
    generate_dataset(10)
