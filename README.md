# SEC IAPD Phase 1 Extractor

Compliance-focused data extractor for public SEC IAPD records, with optional public website email lookup.

## What it does

- Uses the SEC IAPD public compilation feed (`IA_FIRM_SEC_Feed_*.xml.gz`)
- Opens each selected firm's public IAPD summary profile URL (`https://adviserinfo.sec.gov/firm/summary/{FirmCRD}`)
- Filters to:
  - `FirmType=Registered` (Investment Adviser)
  - `Status=APPROVED*`
  - U.S.-based main office
  - Valid U.S. state code
  - AUM (`Q5F2C`) between configured bounds
  - Total employees (`TtlEmp`) between configured bounds
- Builds CSV rows for two contact roles per firm:
  - `Chief Compliance Officer`
  - `Executive Officer`
- If email is not present in SEC feed context, crawls only public firm website pages and extracts visible emails
- Prioritizes the firm Contact page for website email extraction
- Skips social profile URLs as primary firm websites (e.g., LinkedIn/Facebook)
- Uses retry + backoff for network resilience (`429/5xx` handling)

## Output schema

`firm_name,sec_number,aum,employees,city,state,website,contact_name,contact_email,phone,source`

Missing values are written as `N/A`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python sec_iapd_scraper.py --out advisors.csv
```

Useful options:

```bash
python sec_iapd_scraper.py --max-firms 100 --crawl-max-pages 8 --crawl-delay 0.5 --request-timeout 45 --max-runtime-seconds 120 --out advisors_2min.csv
```

SEC-only mode (no firm website crawling):

```bash
python sec_iapd_scraper.py --disable-website-crawl --out advisors_sec_only.csv
```

If you need to skip opening IAPD profile pages and rely only on the SEC feed, add:

```bash
python sec_iapd_scraper.py --allow-feed-only --out advisors_feed_only.csv
```

Quick verification run:

```bash
python sec_iapd_scraper.py --max-firms 5 --crawl-max-pages 6 --crawl-delay 0.3 --out advisors_sample.csv
```

## Notes

- Website crawling is public-page only; no login or bypass behavior.
- For contact names, if no reliable role-matched name is found, `contact_name` defaults to the role label.
