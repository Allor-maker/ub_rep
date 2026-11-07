"""Independent reference calculations for public test datasets."""

import csv
import io
import re
from fractions import Fraction


def public_reference(instruction, contents):
    """Independent formulas for the local public datasets; never read labels."""
    if instruction.startswith("A warehouse starts with"):
        initial = int(re.search(r"starts with (\d+) boxes", instruction)[1])
        if "unspecified" in instruction:
            return None
        if "deliveries" in instruction:
            shipment, days = map(
                int, re.search(r"ships (\d+) boxes per day for (\d+) days", instruction).groups()
            )
            deliveries, quantity = map(
                int, re.search(r"receives (\d+) deliveries of (\d+) boxes", instruction).groups()
            )
            return str(initial - shipment * days + deliveries * quantity)
        if "damaged" in instruction:
            received = int(re.search(r"receives (\d+) new", instruction)[1])
            damaged = int(re.search(r"discards (\d+) damaged", instruction)[1])
            batches, quantity = map(
                int, re.search(r"sends (\d+) batches of (\d+) boxes", instruction).groups()
            )
            return str(initial + received - damaged - batches * quantity)
        if "extra" in instruction:
            shipment, days = map(
                int, re.search(r"ships (\d+) boxes per day for (\d+) days", instruction).groups()
            )
            sites = int(re.search(r"among (\d+) sites", instruction)[1])
            extra = int(re.search(r"extra (\d+) boxes", instruction)[1])
            return str((initial - shipment * days) // sites + extra)
        splits = list(map(int, re.findall(r"split(?: equally)? into (\d+)", instruction)))
        if len(splits) == 3 and instruction.count("doubled") == 2:
            return str(((initial // splits[0]) * 2 // splits[1]) * 2 // splits[2])
        return None
    if "knowledge base" in instruction:
        profiles = {}
        for name, text in contents.items():
            if not name.endswith(".md"):
                continue
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            profiles[lines[0][2:]] = dict(line.split(": ", 1) for line in lines[1:])
        if "headquarters of" in instruction:
            person = re.search(r"headquarters of (.+?)'s employer", instruction)[1]
            return profiles[profiles[person]["Employer"]]["Headquarters"]
        if "employer founded" in instruction:
            person = re.search(r"was (.+?)'s employer founded", instruction)[1]
            return profiles[profiles[person]["Employer"]]["Founded"]
        if "mentor born" in instruction:
            person = re.search(r"was (.+?)'s mentor born", instruction)[1]
            return profiles[profiles[person]["Mentor"]]["Born"]
        match = re.search(r"In which year was ([^?]+?) born\?", instruction)
        return profiles[match[1]]["Born"] if match else None
    if "has columns" in instruction:
        if "sales tax" in instruction:
            return None  # Public schema has no tax information.
        path = re.search(r"The file (\S+\.csv)", instruction)[1]
        rows = list(csv.DictReader(io.StringIO(contents[path], newline="")))
        product = re.search(r"product '([^']+)'", instruction)[1]
        region = re.search(r"region '([^']+)'", instruction)
        months = re.search(r"months (\d+) through (\d+)", instruction)
        required = ["product"] + (["region"] if region else []) + (["month"] if months else [])
        if any(not (row.get(column) or "").strip() for row in rows for column in required):
            return None
        if "How many orders" in instruction:
            # The hand-written oracle refuses missing IDs and incompatible identity/region data.
            known_ids, known_regions = {}, {}
            for row in rows:
                order_id = row.get("order_id") or ""
                if not order_id.strip() or order_id != order_id.strip():
                    return None
                folded = order_id.casefold()
                if folded in known_ids and known_ids[folded] != order_id:
                    return None
                known_ids[folded] = order_id
                if region:
                    if order_id in known_regions and known_regions[order_id] != row["region"]:
                        return None
                    known_regions[order_id] = row["region"]
        selected = [
            r
            for r in rows
            if r["product"] == product
            and (region is None or r["region"] == region[1])
            and (months is None or int(months[1]) <= int(r["month"]) <= int(months[2]))
        ]
        if "How many orders" in instruction:
            # The externally computed oracle counts order identities, not CSV records.
            return str(len({row["order_id"] for row in selected}))
        value = sum(
            (
                Fraction(r["units"])
                * (Fraction(r["price"]) if "total revenue" in instruction else 1)
                for r in selected
            ),
            Fraction(0),
        )
        assert value.denominator == 1
        return str(value.numerator)
    return None
