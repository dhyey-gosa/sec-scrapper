#!/usr/bin/env python3
import argparse
import csv
import gzip
import re
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MANIFEST_URL = "https://reports.adviserinfo.sec.gov/reports/CompilationReports/CompilationReports.manifest.json"
REPORT_BASE_URL = "https://reports.adviserinfo.sec.gov/reports/CompilationReports/"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HREF_RE = re.compile(r"href\s*=\s*[\"']([^\"'#]+)[\"']", re.IGNORECASE)
MAILTO_RE = re.compile(r"mailto:([^\"'?#\s>]+)", re.IGNORECASE)

US_STATE_CODES = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

SOCIAL_DOMAINS = {
    "facebook.com",
    "linkedin.com",
    "youtube.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
}

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
    firm_crd: str
    aum_value: int
    employees_value: int
    city: str
    state: str
    main_address: str
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


def safe_urlparse(url: str):
    try:
        return urlparse(url)
    except Exception:
        return None


def create_session(request_timeout: int) -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "RIAResearchBot/1.1 (+public SEC data market research; no login bypass)",
            "Accept": "application/json, text/html, application/xhtml+xml, */*",
        }
    )
    session.request_timeout = request_timeout  # type: ignore[attr-defined]
    return session


def session_timeout(session: requests.Session, fallback: int = 60) -> int:
    value = getattr(session, "request_timeout", fallback)
    try:
        return int(value)
    except Exception:
        return fallback


def is_social_domain(netloc: str) -> bool:
    netloc = (netloc or "").lower().strip()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return any(netloc == d or netloc.endswith(f".{d}") for d in SOCIAL_DOMAINS)


def pick_primary_website(urls: List[str]) -> str:
    cleaned = [normalize_url(u) for u in urls if (u or "").strip()]
    if not cleaned:
        return ""

    preferred = []
    for u in cleaned:
        parsed = safe_urlparse(u)
        if parsed is None or not parsed.netloc:
            continue
        if not is_social_domain(parsed.netloc):
            preferred.append(u)

    candidates = preferred if preferred else cleaned

    def _sort_key(u: str) -> Tuple[int, int]:
        parsed = safe_urlparse(u)
        path_len = len((parsed.path if parsed is not None else "") or "")
        return (path_len, len(u))

    candidates.sort(key=_sort_key)
    return candidates[0]


def get_latest_sec_firm_feed_url(session: requests.Session) -> str:
    manifest = session.get(MANIFEST_URL, timeout=session_timeout(session, 60))
    manifest.raise_for_status()
    data = manifest.json()
    for item in data.get("files", []):
        name = item.get("name", "")
        if name.startswith("IA_FIRM_SEC_Feed_") and name.endswith(".xml.gz"):
            return REPORT_BASE_URL + name
    raise RuntimeError("Could not locate IA_FIRM_SEC feed in manifest.")


def load_firms_xml(session: requests.Session, feed_url: str) -> ET.Element:
    response = session.get(feed_url, timeout=max(120, session_timeout(session, 180)))
    response.raise_for_status()
    xml_bytes = gzip.decompress(response.content)
    return ET.fromstring(xml_bytes)


def extract_first_website(firm: ET.Element) -> str:
    urls: List[str] = []
    for web in firm.findall("./FormInfo/Part1A/Item1/WebAddrs/WebAddr"):
        text = (web.text or "").strip()
        if text:
            urls.append(text)
    return pick_primary_website(urls)


def compose_main_address(addr: ET.Element) -> str:
    parts = [
        (addr.attrib.get("Strt1") or "").strip(),
        (addr.attrib.get("Strt2") or "").strip(),
        (addr.attrib.get("City") or "").strip(),
        (addr.attrib.get("State") or "").strip().upper(),
        (addr.attrib.get("PostlCd") or "").strip(),
        (addr.attrib.get("Cntry") or "").strip(),
    ]
    return ", ".join([p for p in parts if p])


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
    firm_crd = (info.attrib.get("FirmCrdNb") or "").strip()
    firm_name = (info.attrib.get("BusNm") or info.attrib.get("LegalNm") or "").strip()
    city = (addr.attrib.get("City") or "").strip()
    state = (addr.attrib.get("State") or "").strip().upper()
    phone = (addr.attrib.get("PhNb") or "").strip()
    website = extract_first_website(firm)
    main_address = compose_main_address(addr)

    if not sec_number or not firm_name or state not in US_STATE_CODES:
        return None

    return FirmRecord(
        firm_name=firm_name,
        sec_number=sec_number,
        firm_crd=firm_crd,
        aum_value=aum,
        employees_value=employees,
        city=city,
        state=state,
        main_address=main_address,
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
    return {e.lower() for e in possible if is_public_email(e.lower())}


def extract_visible_emails(text: str, html: str) -> List[str]:
    visible = [e.lower() for e in EMAIL_RE.findall(text)]
    mailto = [e.lower() for e in MAILTO_RE.findall(html)]
    emails = [e for e in (visible + mailto) if is_public_email(e)]
    deduped: List[str] = []
    seen: Set[str] = set()
    for e in emails:
        if e not in seen:
            seen.add(e)
            deduped.append(e)
    return deduped


def is_public_email(email: str) -> bool:
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False
    local = email.split("@", 1)[0]
    if any(token in local for token in ["noreply", "no-reply", "donotreply", "example"]):
        return False
    if email.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        return False
    return True


def clean_html(html: str) -> str:
    html = re.sub(r"<!--([\s\S]*?)-->", " ", html)
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    html = re.sub(r"\b[A-Za-z0-9._%+-]+\s*\[at\]\s*[A-Za-z0-9.-]+\s*\[dot\]\s*[A-Za-z]{2,}\b", " ", html)
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
        candidate_name = sanitize_candidate_name(match.group(1).strip())
        if is_probably_person_name(candidate_name):
            selected_name = candidate_name
        window_start = max(0, match.start() - 220)
        window_end = min(len(text), match.end() + 220)
        window = text[window_start:window_end]
        emails = EMAIL_RE.findall(window)
        if emails:
            best = next((e.lower() for e in emails if is_public_email(e.lower())), None)
            selected_email = best
        break

    return selected_name, selected_email


def is_probably_person_name(value: str) -> bool:
    value = (value or "").strip()
    if not value:
        return False
    if any(ch.isdigit() for ch in value):
        return False
    words = value.split()
    if len(words) < 2 or len(words) > 4:
        return False
    lowered = value.lower()
    for token in [
        "llc",
        "inc",
        "advisors",
        "capital",
        "management",
        "wealth",
        "partners",
        "portfolio",
        "manager",
        "officer",
    ]:
        if token in lowered:
            return False
    return all(len(w) >= 2 for w in words)


def sanitize_candidate_name(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    value = re.sub(
        r"^(chief|executive|compliance|portfolio|managing|senior|principal|officer|partner|director|president)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip(" ,|-:")


def build_robot_parser(session: requests.Session, website: str) -> Optional[RobotFileParser]:
    try:
        parsed = urlparse(website)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = session.get(robots_url, timeout=min(session_timeout(session), 20))
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
        if p.path.lower().endswith((".pdf", ".doc", ".docx", ".xls", ".xlsx")):
            continue
        out.append(p._replace(fragment="", query="").geturl())
    return out


def extract_contact_links(base_url: str, html: str, domain: str) -> List[str]:
    links = extract_links(base_url, html, domain)
    filtered: List[str] = []
    for u in links:
        parsed = safe_urlparse(u)
        path = ((parsed.path if parsed is not None else "") or "").lower()
        if any(token in path for token in ["contact", "contact-us", "contactus"]):
            filtered.append(u)
    seen: Set[str] = set()
    ordered: List[str] = []
    for u in filtered:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


def score_url(url: str) -> int:
    parsed = safe_urlparse(url)
    path = ((parsed.path if parsed is not None else "") or "").lower()
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
    parsed_website = safe_urlparse(website)
    if parsed_website is None or not parsed_website.netloc:
        return result
    if is_social_domain(parsed_website.netloc):
        return result

    parsed = parsed_website
    domain = parsed.netloc
    start_url = f"{parsed.scheme}://{domain}/"

    robots = build_robot_parser(session, start_url)
    queue = deque([start_url])
    visited: Set[str] = set()
    pages = 0
    contact_seeded = False

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
            resp = session.get(url, timeout=min(session_timeout(session), 25))
        except Exception:
            continue

        pages += 1
        time.sleep(delay_seconds)

        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type:
            continue

        html = resp.text
        text = clean_html(html)
        all_emails = extract_visible_emails(text, html)
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

        if not contact_seeded:
            contact_links = extract_contact_links(url, html, domain)
            for link in reversed(contact_links):
                if link not in visited:
                    queue.appendleft(link)
            if contact_links:
                contact_seeded = True

        links = extract_links(url, html, domain)
        links.sort(key=score_url, reverse=True)
        for link in links:
            if link not in visited:
                queue.append(link)

        if all(result[r]["email"] for r in ["Chief Compliance Officer", "Executive Officer"]):
            break

    return result


def open_iapd_firm_profile(session: requests.Session, firm_crd: str) -> bool:
    firm_crd = (firm_crd or "").strip()
    if not firm_crd:
        return False
    url = f"https://adviserinfo.sec.gov/firm/summary/{firm_crd}"
    try:
        resp = session.get(url, timeout=min(session_timeout(session), 25))
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    body = (resp.text or "")[:5000].lower()
    return "investment adviser public disclosure" in body or "iapd" in body


def compile_rows(
    firms: List[FirmRecord],
    session: requests.Session,
    max_firms: Optional[int],
    crawl_max_pages: Optional[int],
    crawl_delay: float,
    disable_website_crawl: bool,
    max_runtime_seconds: Optional[int],
    require_profile_open: bool,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_sec: Set[str] = set()
    website_cache: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    profile_open_cache: Dict[str, bool] = {}
    started = time.time()

    for firm in firms:
        if max_runtime_seconds is not None and time.time() - started >= max_runtime_seconds:
            break
        if firm.sec_number in seen_sec:
            continue
        seen_sec.add(firm.sec_number)

        sec_emails = extract_possible_sec_emails(firm.source_xml)

        if firm.firm_crd not in profile_open_cache:
            profile_open_cache[firm.firm_crd] = open_iapd_firm_profile(session, firm.firm_crd)
        if require_profile_open and not profile_open_cache[firm.firm_crd]:
            continue
        website_contacts: Dict[str, Dict[str, Optional[str]]] = {
            "Chief Compliance Officer": {"name": None, "email": None},
            "Executive Officer": {"name": None, "email": None},
            "generic": {"name": None, "email": None},
        }

        if firm.website and not disable_website_crawl:
            parsed_website = safe_urlparse(firm.website)
            host = (parsed_website.netloc.lower() if parsed_website is not None else "")
            if not host:
                continue
            if host in website_cache:
                website_contacts = website_cache[host]
            else:
                website_contacts = crawl_website_for_contacts(
                    session=session,
                    website=firm.website,
                    max_pages=crawl_max_pages,
                    delay_seconds=crawl_delay,
                )
                website_cache[host] = website_contacts

        for role in ["Chief Compliance Officer", "Executive Officer"]:
            contact_name = website_contacts[role]["name"] if is_probably_person_name(website_contacts[role]["name"] or "") else role
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
        "--disable-website-crawl",
        action="store_true",
        help="Use SEC feed only and skip firm website crawling",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=60,
        help="Per-request timeout (seconds) for SEC and website HTTP requests",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=None,
        help="Optional total runtime cap for row compilation",
    )
    parser.add_argument(
        "--allow-feed-only",
        action="store_true",
        help="Do not require opening the public IAPD firm profile page for each selected firm",
    )
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

    session = create_session(args.request_timeout)

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
        disable_website_crawl=args.disable_website_crawl,
        max_runtime_seconds=args.max_runtime_seconds,
        require_profile_open=(not args.allow_feed_only),
    )

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    firm_count = len({r["sec_number"] for r in rows})
    print(f"Wrote {len(rows)} rows ({firm_count} firms) to {args.out}")


if __name__ == "__main__":
    main()
