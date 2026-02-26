#!/usr/bin/env python3
import argparse
import csv
import gzip
import io
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests


MANIFEST_URL = "https://reports.adviserinfo.sec.gov/reports/CompilationReports/CompilationReports.manifest.json"
REPORT_BASE_URL = "https://reports.adviserinfo.sec.gov/reports/CompilationReports/"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HREF_RE = re.compile(r"href\s*=\s*[\"']([^\"'#]+)[\"']", re.IGNORECASE)

CSV_COLUMNS = [
    "firm_name",
    "sec_number",
    "aum",
    "employees",
    "city",
    "state",
    "website",
    "contact_name",
    "contact_email",
    "phone",
    "source",
]

ROLE_KEYWORDS = {
    "Chief Compliance Officer": [
        "chief compliance officer",
        "cco",
        "compliance officer",
    ],
    "Executive Officer": [
        "executive officer",
        "chief executive officer",
        "ceo",
        "president",
        "managing partner",
    ],
}


@dataclass
class FirmRecord:
    firm_name: str
    sec_number: str
    aum_value: int
    employees_value: int
    city: str
    state: str
    website: str
    phone: str
    source_xml: ET.Element


def to_na(value: Optional[str]) -> str:
    if value is None:
        return "N/A"
    value = str(value).strip()
    return value if value else "N/A"


def to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = re.sub(r"[^0-9]", "", str(value))
    if not value:
        return None
    return int(value)


def as_money_string(amount: Optional[int]) -> str:
    if amount is None:
        return "N/A"
    return f"${amount:,}"


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    return url


def get_latest_sec_firm_feed_url(session: requests.Session) -> str:
    manifest = session.get(MANIFEST_URL, timeout=60)
    manifest.raise_for_status()
    data = manifest.json()
    for item in data.get("files", []):
        name = item.get("name", "")
        if name.startswith("IA_FIRM_SEC_Feed_") and name.endswith(".xml.gz"):
            return REPORT_BASE_URL + name
    raise RuntimeError("Could not locate IA_FIRM_SEC feed in manifest.")


def load_firms_xml(session: requests.Session, feed_url: str) -> ET.Element:
    response = session.get(feed_url, timeout=180)
    response.raise_for_status()
    xml_bytes = gzip.decompress(response.content)
    return ET.fromstring(xml_bytes)


def extract_first_website(firm: ET.Element) -> str:
    webs = firm.findall("./FormInfo/Part1A/Item1/WebAddrs/WebAddr")
    for web in webs:
        text = (web.text or "").strip()
        if text:
            return normalize_url(text)
    return ""


def extract_firm_record(firm: ET.Element) -> Optional[FirmRecord]:
    info = firm.find("Info")
    rgstn = firm.find("Rgstn")
    addr = firm.find("MainAddr")
    item5a = firm.find("./FormInfo/Part1A/Item5A")
    item5f = firm.find("./FormInfo/Part1A/Item5F")

    if info is None or rgstn is None or addr is None or item5a is None or item5f is None:
        return None

    firm_type = rgstn.attrib.get("FirmType", "")
    status = rgstn.attrib.get("St", "")
    country = addr.attrib.get("Cntry", "")

    if firm_type != "Registered":
        return None
    if not status.startswith("APPROVED"):
        return None
    if country.lower() not in {"united states", "usa", "us", "u.s.", "u.s.a."}:
        return None

    employees = to_int(item5a.attrib.get("TtlEmp"))
    aum = to_int(item5f.attrib.get("Q5F2C"))
    if employees is None or aum is None:
        return None

    sec_number = (info.attrib.get("SECNb") or "").strip()
    firm_name = (info.attrib.get("BusNm") or info.attrib.get("LegalNm") or "").strip()
    city = (addr.attrib.get("City") or "").strip()
    state = (addr.attrib.get("State") or "").strip().upper()
    phone = (addr.attrib.get("PhNb") or "").strip()
    website = extract_first_website(firm)

    if not sec_number or not firm_name:
        return None

    return FirmRecord(
        firm_name=firm_name,
        sec_number=sec_number,
        aum_value=aum,
        employees_value=employees,
        city=city,
        state=state,
        website=website,
        phone=phone,
        source_xml=firm,
    )


def extract_possible_sec_emails(firm: ET.Element) -> Set[str]:
    possible: Set[str] = set()
    for web in firm.findall("./FormInfo/Part1A/Item1/WebAddrs/WebAddr"):
        text = (web.text or "").strip()
        if "@" in text:
            possible.update(EMAIL_RE.findall(text))
    return {e.lower() for e in possible}


def clean_html(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\s+", " ", html)
    return html


def role_candidates_from_text(text: str, role: str) -> Tuple[Optional[str], Optional[str]]:
    role_kw = "|".join(re.escape(x) for x in ROLE_KEYWORDS[role])
    patterns = [
        re.compile(
            rf"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,3}})\b\s*(?:,|\-|\|)\s*(?:{role_kw})\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"(?:{role_kw})\b\s*(?:,|\-|:|\|)\s*\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{1,3}})\b",
            re.IGNORECASE,
        ),
    ]

    selected_name: Optional[str] = None
    selected_email: Optional[str] = None
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        selected_name = match.group(1).strip()
        window_start = max(0, match.start() - 220)
        window_end = min(len(text), match.end() + 220)
        window = text[window_start:window_end]
        emails = EMAIL_RE.findall(window)
        if emails:
            selected_email = emails[0].lower()
        break

    return selected_name, selected_email


def build_robot_parser(session: requests.Session, website: str) -> Optional[RobotFileParser]:
    try:
        parsed = urlparse(website)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = session.get(robots_url, timeout=20)
        if resp.status_code != 200:
            return None
        rp = RobotFileParser()
        rp.parse(resp.text.splitlines())
        return rp
    except Exception:
        return None


def extract_links(base_url: str, html: str, domain: str) -> List[str]:
    out: List[str] = []
    for href in HREF_RE.findall(html):
        absolute = urljoin(base_url, href)
        p = urlparse(absolute)
        if p.scheme not in {"http", "https"}:
            continue
        if p.netloc != domain:
            continue
        out.append(p._replace(fragment="", query="").geturl())
    return out


def score_url(url: str) -> int:
    path = (urlparse(url).path or "").lower()
    score = 0
    for token, value in [
        ("contact", 5),
        ("team", 4),
        ("about", 3),
        ("leadership", 4),
        ("compliance", 4),
        ("people", 3),
    ]:
        if token in path:
            score += value
    return score


def crawl_website_for_contacts(
    session: requests.Session,
    website: str,
    max_pages: Optional[int],
    delay_seconds: float,
) -> Dict[str, Dict[str, Optional[str]]]:
    result: Dict[str, Dict[str, Optional[str]]] = {
        "Chief Compliance Officer": {"name": None, "email": None},
        "Executive Officer": {"name": None, "email": None},
        "generic": {"name": None, "email": None},
    }

    website = normalize_url(website)
    if not website:
        return result

    parsed = urlparse(website)
    domain = parsed.netloc
    start_url = f"{parsed.scheme}://{domain}/"

    robots = build_robot_parser(session, start_url)
    queue = deque([start_url])
    visited: Set[str] = set()
    pages = 0

    while queue:
        if max_pages is not None and pages >= max_pages:
            break
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        user_agent = str(session.headers.get("User-Agent", "*"))
        if robots and not robots.can_fetch(user_agent, url):
            continue

        try:
            resp = session.get(url, timeout=20)
        except Exception:
            continue

        pages += 1
        time.sleep(delay_seconds)

        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            continue

        html = resp.text
        text = clean_html(html)
        all_emails = [e.lower() for e in EMAIL_RE.findall(text + " " + html)]
        if all_emails and result["generic"]["email"] is None:
            result["generic"]["email"] = all_emails[0]

        for role in ["Chief Compliance Officer", "Executive Officer"]:
            if result[role]["email"] and result[role]["name"]:
                continue
            name, email = role_candidates_from_text(text, role)
            if name and result[role]["name"] is None:
                result[role]["name"] = name
            if email and result[role]["email"] is None:
                result[role]["email"] = email

        links = extract_links(url, html, domain)
        links.sort(key=score_url, reverse=True)
        for link in links:
            if link not in visited:
                queue.append(link)

        if all(result[r]["email"] for r in ["Chief Compliance Officer", "Executive Officer"]):
            break

    return result


def compile_rows(
    firms: List[FirmRecord],
    session: requests.Session,
    max_firms: Optional[int],
    crawl_max_pages: Optional[int],
    crawl_delay: float,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_sec: Set[str] = set()

    for firm in firms:
        if firm.sec_number in seen_sec:
            continue
        seen_sec.add(firm.sec_number)

        sec_emails = extract_possible_sec_emails(firm.source_xml)
        website_contacts: Dict[str, Dict[str, Optional[str]]] = {
            "Chief Compliance Officer": {"name": None, "email": None},
            "Executive Officer": {"name": None, "email": None},
            "generic": {"name": None, "email": None},
        }

        if firm.website:
            website_contacts = crawl_website_for_contacts(
                session=session,
                website=firm.website,
                max_pages=crawl_max_pages,
                delay_seconds=crawl_delay,
            )

        for role in ["Chief Compliance Officer", "Executive Officer"]:
            contact_name = website_contacts[role]["name"] or role
            contact_email = None
            source = "SEC_IAPD"

            if sec_emails:
                contact_email = sorted(sec_emails)[0]
                source = "SEC_IAPD"
            elif website_contacts[role]["email"]:
                contact_email = website_contacts[role]["email"]
                source = "FIRM_WEBSITE"
            elif website_contacts["generic"]["email"]:
                contact_email = website_contacts["generic"]["email"]
                source = "FIRM_WEBSITE"

            rows.append(
                {
                    "firm_name": to_na(firm.firm_name),
                    "sec_number": to_na(firm.sec_number),
                    "aum": to_na(as_money_string(firm.aum_value)),
                    "employees": to_na(str(firm.employees_value)),
                    "city": to_na(firm.city),
                    "state": to_na(firm.state),
                    "website": to_na(firm.website),
                    "contact_name": to_na(contact_name),
                    "contact_email": to_na(contact_email),
                    "phone": to_na(firm.phone),
                    "source": source,
                }
            )

        if max_firms is not None and len(seen_sec) >= max_firms:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC IAPD Phase 1 extractor for small U.S. RIAs (compliance-focused)."
    )
    parser.add_argument("--out", default="advisors.csv", help="Output CSV path")
    parser.add_argument("--min-aum", type=int, default=50_000_000)
    parser.add_argument("--max-aum", type=int, default=300_000_000)
    parser.add_argument("--min-employees", type=int, default=1)
    parser.add_argument("--max-employees", type=int, default=10)
    parser.add_argument("--max-firms", type=int, default=None, help="Optional cap for quick test runs")
    parser.add_argument(
        "--crawl-max-pages",
        type=int,
        default=None,
        help="Optional cap for website crawling pages per domain",
    )
    parser.add_argument(
        "--crawl-delay",
        type=float,
        default=0.7,
        help="Delay between website crawl requests in seconds",
    )
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "RIAResearchBot/1.0 (+public SEC data market research; no login bypass)",
            "Accept": "application/json, text/html, application/xhtml+xml, */*",
        }
    )

    feed_url = get_latest_sec_firm_feed_url(session)
    root = load_firms_xml(session, feed_url)

    all_firms: List[FirmRecord] = []
    for firm_el in root.findall(".//Firm"):
        record = extract_firm_record(firm_el)
        if not record:
            continue
        if not (args.min_aum <= record.aum_value <= args.max_aum):
            continue
        if not (args.min_employees <= record.employees_value <= args.max_employees):
            continue
        all_firms.append(record)

    rows = compile_rows(
        firms=all_firms,
        session=session,
        max_firms=args.max_firms,
        crawl_max_pages=args.crawl_max_pages,
        crawl_delay=args.crawl_delay,
    )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    firm_count = len({r["sec_number"] for r in rows})
    print(f"Wrote {len(rows)} rows ({firm_count} firms) to {args.out}")


if __name__ == "__main__":
    main()
