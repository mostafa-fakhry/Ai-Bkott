"""
BKOTT Profile Review Agent — Multi-Profile
Handles: Freelancer (85 pts) | Company (96 pts) | CV/Job Seeker (105 pts)

POST /review-profile  →  accepts any profile type via `profile_type` field
"""

import os
import re
import json
from typing import Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import httpx

# Load variables from a local .env file (no-op if the file is absent)
load_dotenv()

app = FastAPI(
    title="BKOTT Profile Review Agent",
    description="AI-powered multi-profile reviewer — Freelancer · Company · CV",
    version="2.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST", "GET"], allow_headers=["*"])

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL      = "claude-sonnet-4-6"


# ══════════════════════════════════════════════════════════════
#  ENUMS
# ══════════════════════════════════════════════════════════════

class ProfileType(str, Enum):
    freelancer = "freelancer"
    company    = "company"
    cv         = "cv"


class DecisionStatus(str, Enum):
    approve         = "approve"
    reject          = "reject"
    flag_for_human  = "flag_for_human"


# ══════════════════════════════════════════════════════════════
#  SHARED SUB-MODELS
# ══════════════════════════════════════════════════════════════

class PortfolioItem(BaseModel):
    title:       str
    description: Optional[str] = None
    link:        Optional[str] = None
    media_urls:  list[str]     = Field(default_factory=list)


class WorkExperience(BaseModel):
    company:    str
    role:       str
    start_year: int
    end_year:   Optional[int] = None   # None = present
    description: Optional[str] = None


class Education(BaseModel):
    institution: str
    degree:      str
    field:       Optional[str] = None
    year:        Optional[int] = None


class FAQItem(BaseModel):
    question: str
    answer:   str


class DeliveryGovernorate(BaseModel):
    country:       str
    governorate:   str
    delivery_fee:  Optional[float] = None   # None = negotiated


# ══════════════════════════════════════════════════════════════
#  PROFILE INPUT SCHEMAS  (one per type)
# ══════════════════════════════════════════════════════════════

# ── FREELANCER  (85 pts) ─────────────────────────────────────
class FreelancerProfile(BaseModel):
    # Step 1 — Basic data                     [Critical]
    name:                   str
    country:                str
    primary_skill_category: str
    photo_url:              Optional[str] = None
    headline:               str

    # Step 2 — Skills & experience             [Important]
    skills:              list[str]
    years_of_experience: int = Field(..., ge=0, le=60)
    languages:           list[str]

    # Step 3 — Portfolio                       [Important]
    portfolio: list[PortfolioItem] = Field(default_factory=list)
    bio:       Optional[str]       = None

    # Optional enrichment
    hourly_rate_min:     Optional[float] = None
    hourly_rate_max:     Optional[float] = None
    availability_status: Optional[str]   = None
    working_hours:       Optional[str]   = None
    response_time:       Optional[str]   = None

    # Contact fields (allowed here — rejected only if in description text)
    contact_email:  Optional[str] = None
    contact_phone:  Optional[str] = None


# ── COMPANY  (96 pts) ────────────────────────────────────────
class CompanyProfile(BaseModel):
    # Step 1 — Basic data                      [Critical]
    company_name:     str
    country:          str
    governorate:      Optional[str] = None
    primary_category: str
    description:      str
    logo_url:         Optional[str] = None

    # Step 2 — Business type                   [Critical]
    business_type:    str   # retail / wholesale / services / mixed
    target_market:    str   # B2C / B2B / both
    presence:         str   # physical / online / both

    # Step 3 — Page content                    [Important]
    cover_banner_url: Optional[str]       = None
    gallery_urls:     list[str]           = Field(default_factory=list)
    working_hours:    Optional[str]       = None
    social_links:     dict[str, str]      = Field(default_factory=dict)
    faq:              list[FAQItem]       = Field(default_factory=list)
    serving_areas:    list[DeliveryGovernorate] = Field(default_factory=list)

    # Optional enrichment
    founded_year:     Optional[int]  = None
    team_size:        Optional[str]  = None   # e.g. "1-10"
    contact_email:    Optional[str]  = None
    contact_phone:    Optional[str]  = None
    website_url:      Optional[str]  = None


# ── CV / JOB SEEKER  (105 pts) ───────────────────────────────
class CVProfile(BaseModel):
    # Step 1 — Personal info                   [Critical]
    name:        str
    country:     str
    governorate: Optional[str] = None
    photo_url:   Optional[str] = None

    # Contact (allowed in their own fields)
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None

    # Step 2 — Job intent                      [Critical]
    desired_role:    str
    industry:        str
    employment_type: str   # full-time / part-time / contract
    salary_min:      Optional[float] = None
    salary_max:      Optional[float] = None
    willing_to_relocate: bool = False

    # Step 3 — Experience                      [Important]
    work_history:   list[WorkExperience] = Field(default_factory=list)
    education:      list[Education]      = Field(default_factory=list)
    certifications: list[str]            = Field(default_factory=list)
    languages:      list[str]            = Field(default_factory=list)
    skills:         list[str]            = Field(default_factory=list)
    bio:            Optional[str]        = None


# ── UNIFIED SUBMISSION WRAPPER ────────────────────────────────
class ProfileSubmission(BaseModel):
    profile_type: ProfileType
    data: dict[str, Any]   # raw dict; we parse into the correct model below


# ══════════════════════════════════════════════════════════════
#  OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════

class FieldFeedback(BaseModel):
    field:            str
    tier:             str   # critical / important / optional
    status:           str   # ok / warning / fail
    message:          str
    points_awarded:   int
    points_possible:  int


class ReviewDecision(BaseModel):
    profile_type:         str
    decision:             str
    score:                int
    max_score:            int
    score_breakdown:      dict
    field_feedback:       list[FieldFeedback]
    auto_reject_triggers: list[str]
    summary:              str
    applicant_message:    Optional[str] = None   # Arabic-first if reject/flag


# ══════════════════════════════════════════════════════════════
#  CONTENT GUARD — shared across all profile types
#  Scans FREE-TEXT FIELDS ONLY (not dedicated contact fields)
# ══════════════════════════════════════════════════════════════

# Offensive / profanity patterns (Arabic + English)
PROFANITY_PATTERNS = [
    # Arabic offensive terms (transliterated + Arabic script)
    r'\bكس\b', r'\bزب\b', r'\bنيك\b', r'\bشرموط', r'\bعرص', r'\bمنيوك',
    r'\bكلب\b', r'\bحمار\b', r'\bعاهر', r'\bقحب', r'\bلعين\b',
    r'\bيلعن\b', r'\bابن.*حرام', r'\bبنت.*حرام',
    # English offensive
    r'\bfuck', r'\bshit\b', r'\bbitch\b', r'\basshole', r'\bcunt\b',
    r'\bdick\b', r'\bpussy\b', r'\bwhore\b', r'\bslut\b',
]

# Contact info patterns — only trigger when found in FREE TEXT, not in contact fields
CONTACT_IN_TEXT_PATTERNS = [
    # Email-like
    r'[a-zA-Z0-9._%+\-]+\s*@\s*[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
    # Phone numbers (local + international formats)
    r'(?:\+?\d[\d\s\-().]{7,}\d)',
    # Social media handles / platform mentions
    r'@[a-zA-Z0-9_\.]{2,}',
    r'\bwhatsapp\b', r'\bواتساب\b', r'\bwa\.me\b',
    r'\btelegram\b', r'\bتيليجرام\b', r'\bt\.me\b',
    r'\binstagram\b', r'\bانستا\b', r'\binsta\b',
    r'\bfacebook\b', r'\bفيسبوك\b', r'\bfb\.com\b',
    r'\btiktok\b', r'\bتيك تو\b',
    r'\bsnapchat\b', r'\bسناب\b',
    r'\btwitter\b', r'\bتويتر\b', r'\bx\.com\b',
    r'\blinkedin\b', r'\bلينكدإن\b',
    r'\bيوتيوب\b', r'\byoutube\b',
]

PLACEHOLDER_PATTERNS = [
    r'\blorem ipsum\b', r'\btest\b', r'\basdf\b', r'\bqwerty\b',
    r'\bexample\b', r'\bsample\b', r'\bplaceholder\b',
    r'\baختبار\b', r'\bتجربة\b', r'\bنص تجريبي\b',
]


def collect_free_text_fields(profile_type: str, data: dict) -> list[tuple[str, str]]:
    """
    Returns list of (field_name, text_value) for fields that are
    free-text and should NOT contain contact info or profanity.
    Dedicated contact fields (contact_email, contact_phone) are EXCLUDED.
    """
    if profile_type == "freelancer":
        return [
            ("headline",    data.get("headline", "") or ""),
            ("bio",         data.get("bio", "") or ""),
            *[(f"portfolio[{i}].title",       p.get("title",""))
              for i,p in enumerate(data.get("portfolio",[]))],
            *[(f"portfolio[{i}].description", p.get("description","") or "")
              for i,p in enumerate(data.get("portfolio",[]))],
        ]
    elif profile_type == "company":
        return [
            ("description",  data.get("description", "") or ""),
            *[(f"faq[{i}].answer", f.get("answer",""))
              for i,f in enumerate(data.get("faq",[]))],
        ]
    elif profile_type == "cv":
        return [
            ("bio",           data.get("bio", "") or ""),
            *[(f"work_history[{i}].description", w.get("description","") or "")
              for i,w in enumerate(data.get("work_history",[]))],
        ]
    return []


def scan_content(profile_type: str, data: dict) -> list[str]:
    """
    Returns list of triggered violation strings.
    Any single trigger = immediate REJECT regardless of score.
    """
    triggers = []
    fields = collect_free_text_fields(profile_type, data)
    combined = " ".join(v for _, v in fields).lower()

    for pat in PROFANITY_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE | re.UNICODE):
            triggers.append(f"PROFANITY_OR_OFFENSIVE_LANGUAGE: Pattern '{pat}' matched in free-text fields. Profile rejected.")
            break

    for pat in CONTACT_IN_TEXT_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE | re.UNICODE):
            triggers.append(
                f"CONTACT_INFO_IN_DESCRIPTION: Contact information or social media handle detected in free-text. "
                f"Use the dedicated contact fields instead. Pattern: '{pat}'"
            )
            break

    for pat in PLACEHOLDER_PATTERNS:
        if re.search(pat, combined, re.IGNORECASE | re.UNICODE):
            triggers.append(f"PLACEHOLDER_TEXT: Detected test/placeholder content '{pat}'. Profile appears to be a dummy submission.")
            break

    return triggers


# ══════════════════════════════════════════════════════════════
#  SCORING ENGINES  (one per profile type)
# ══════════════════════════════════════════════════════════════

def _fb(field, tier, awarded, possible, status, message) -> FieldFeedback:
    return FieldFeedback(field=field, tier=tier, status=status,
                         message=message, points_awarded=awarded, points_possible=possible)


# ── FREELANCER — 85 pts ──────────────────────────────────────
#  Critical: name(8) photo(10) skill_category(8) country(4) headline(5) = 35
#  Important: skills(8) experience(5) languages(5) bio(10) portfolio(7)  = 35
#  Optional:  hourly_rate(4) availability(3) working_hours(4) response_time(4) = 15

def score_freelancer(d: FreelancerProfile) -> tuple[dict, list[FieldFeedback]]:
    fb, sc = [], {"critical": 0, "important": 0, "optional": 0}

    def add(field, tier, awarded, possible, status, message):
        sc[tier] += awarded
        fb.append(_fb(field, tier, awarded, possible, status, message))

    # ── Critical ──
    name = d.name.strip()
    if len(name) < 3:
        add("name","critical",0,8,"fail","Name too short (min 3 chars)")
    elif len(name) > 60:
        add("name","critical",5,8,"warning","Name unusually long")
    else:
        add("name","critical",8,8,"ok","Name looks good")

    if d.photo_url:
        add("photo_url","critical",10,10,"ok","Photo provided")
    else:
        add("photo_url","critical",0,10,"fail","Profile photo is required")

    if d.primary_skill_category.strip():
        add("primary_skill_category","critical",8,8,"ok","Skill category set")
    else:
        add("primary_skill_category","critical",0,8,"fail","Primary skill category missing")

    add("country","critical", 4 if d.country.strip() else 0, 4,
        "ok" if d.country.strip() else "fail",
        "Country set" if d.country.strip() else "Country required")

    h = (d.headline or "").strip()
    if len(h) < 10:
        add("headline","critical",0,5,"fail","Headline too short (min 10 chars)")
    elif len(h) > 120:
        add("headline","critical",3,5,"warning","Headline over 120 chars")
    else:
        add("headline","critical",5,5,"ok","Headline length good")

    # ── Important ──
    if len(d.skills) == 0:
        add("skills","important",0,8,"fail","No skills listed")
    elif len(d.skills) < 2:
        add("skills","important",4,8,"warning","Only 1 skill — add more")
    else:
        add("skills","important",8,8,"ok",f"{len(d.skills)} skills listed")

    if d.years_of_experience == 0:
        add("years_of_experience","important",2,5,"warning","0 years — acceptable for entry level")
    else:
        add("years_of_experience","important",5,5,"ok",f"{d.years_of_experience} years experience")

    if len(d.languages) == 0:
        add("languages","important",0,5,"fail","No languages listed")
    else:
        add("languages","important",5,5,"ok",f"{len(d.languages)} language(s)")

    bio_words = len((d.bio or "").split())
    if bio_words == 0:
        add("bio","important",0,10,"fail","Bio is empty")
    elif bio_words < 30:
        add("bio","important",4,10,"warning",f"Bio only {bio_words} words (aim for 50+)")
    elif bio_words < 50:
        add("bio","important",7,10,"warning",f"Bio is {bio_words} words — consider expanding")
    else:
        add("bio","important",10,10,"ok",f"Bio has {bio_words} words")

    n_port = len(d.portfolio)
    if n_port == 0:
        add("portfolio","important",0,7,"fail","No portfolio items")
    elif n_port == 1:
        add("portfolio","important",4,7,"warning","Only 1 portfolio item")
    else:
        add("portfolio","important",7,7,"ok",f"{n_port} portfolio items")

    # ── Optional ──
    if d.hourly_rate_min and d.hourly_rate_max:
        add("hourly_rate","optional",4,4,"ok","Hourly rate range set")
    elif d.hourly_rate_min or d.hourly_rate_max:
        add("hourly_rate","optional",2,4,"warning","Only one bound of rate set")
    else:
        add("hourly_rate","optional",0,4,"ok","Hourly rate not set (optional)")

    add("availability_status","optional",
        3 if d.availability_status else 0, 3,
        "ok", "Availability set" if d.availability_status else "Not set (optional)")

    add("working_hours","optional",
        4 if d.working_hours else 0, 4,
        "ok", "Working hours set" if d.working_hours else "Not set (optional)")

    add("response_time","optional",
        4 if d.response_time else 0, 4,
        "ok", "Response time set" if d.response_time else "Not set (optional)")

    return sc, fb


# ── COMPANY — 96 pts ─────────────────────────────────────────
#  Critical: company_name(8) logo(10) primary_category(8) country(4)
#            description(8) business_type(4) target_market(3) presence(3) = 48
#  Important: cover_banner(6) gallery(6) working_hours(5) serving_areas(8)
#             faq(5) social_links(5)                                        = 35
#  Optional:  founded_year(3) team_size(3) contact_email(3) website_url(4) = 13

def score_company(d: CompanyProfile) -> tuple[dict, list[FieldFeedback]]:
    fb, sc = [], {"critical": 0, "important": 0, "optional": 0}

    def add(field, tier, awarded, possible, status, message):
        sc[tier] += awarded
        fb.append(_fb(field, tier, awarded, possible, status, message))

    # ── Critical ──
    nm = d.company_name.strip()
    if len(nm) < 2:
        add("company_name","critical",0,8,"fail","Company name too short")
    else:
        add("company_name","critical",8,8,"ok","Company name set")

    add("logo_url","critical",
        10 if d.logo_url else 0, 10,
        "ok" if d.logo_url else "fail",
        "Logo provided" if d.logo_url else "Logo required")

    add("primary_category","critical",
        8 if d.primary_category.strip() else 0, 8,
        "ok" if d.primary_category.strip() else "fail",
        "Category set" if d.primary_category.strip() else "Category required")

    add("country","critical",
        4 if d.country.strip() else 0, 4,
        "ok" if d.country.strip() else "fail",
        "Country set" if d.country.strip() else "Country required")

    desc_words = len(d.description.split())
    if desc_words < 20:
        add("description","critical",0,8,"fail",f"Description too short ({desc_words} words, min 20)")
    elif desc_words < 50:
        add("description","critical",5,8,"warning",f"Description short ({desc_words} words)")
    else:
        add("description","critical",8,8,"ok",f"Description has {desc_words} words")

    add("business_type","critical",
        4 if d.business_type.strip() else 0, 4,
        "ok" if d.business_type.strip() else "fail",
        "Business type set" if d.business_type.strip() else "Business type required")

    add("target_market","critical",
        3 if d.target_market.strip() else 0, 3,
        "ok" if d.target_market.strip() else "fail",
        "Target market set" if d.target_market.strip() else "Target market required")

    add("presence","critical",
        3 if d.presence.strip() else 0, 3,
        "ok" if d.presence.strip() else "fail",
        "Presence type set" if d.presence.strip() else "Presence type required")

    # ── Important ──
    add("cover_banner_url","important",
        6 if d.cover_banner_url else 0, 6,
        "ok" if d.cover_banner_url else "warning",
        "Cover banner set" if d.cover_banner_url else "Cover banner missing (improves profile appearance)")

    n_gal = len(d.gallery_urls)
    if n_gal == 0:
        add("gallery_urls","important",0,6,"warning","No gallery images")
    elif n_gal < 3:
        add("gallery_urls","important",3,6,"warning",f"Only {n_gal} gallery image(s) — add more")
    else:
        add("gallery_urls","important",6,6,"ok",f"{n_gal} gallery images")

    add("working_hours","important",
        5 if d.working_hours else 0, 5,
        "ok" if d.working_hours else "warning",
        "Working hours set" if d.working_hours else "Working hours not set")

    n_areas = len(d.serving_areas)
    if n_areas == 0:
        add("serving_areas","important",0,8,"fail","No serving areas configured — buyers can't see delivery scope")
    elif n_areas < 2:
        add("serving_areas","important",4,8,"warning",f"Only {n_areas} serving area")
    else:
        add("serving_areas","important",8,8,"ok",f"{n_areas} serving areas configured")

    n_faq = len(d.faq)
    add("faq","important",
        5 if n_faq >= 2 else (2 if n_faq == 1 else 0), 5,
        "ok" if n_faq >= 2 else ("warning" if n_faq == 1 else "warning"),
        f"{n_faq} FAQ entries" if n_faq else "No FAQ entries")

    n_social = len(d.social_links)
    add("social_links","important",
        5 if n_social >= 1 else 0, 5,
        "ok" if n_social else "warning",
        f"{n_social} social link(s)" if n_social else "No social links")

    # ── Optional ──
    add("founded_year","optional",
        3 if d.founded_year else 0, 3,
        "ok" if d.founded_year else "ok",
        "Founded year set" if d.founded_year else "Not set (optional)")

    add("team_size","optional",
        3 if d.team_size else 0, 3,
        "ok" if d.team_size else "ok",
        "Team size set" if d.team_size else "Not set (optional)")

    add("contact_email","optional",
        3 if d.contact_email else 0, 3,
        "ok" if d.contact_email else "ok",
        "Contact email set" if d.contact_email else "Not set (optional)")

    add("website_url","optional",
        4 if d.website_url else 0, 4,
        "ok" if d.website_url else "ok",
        "Website set" if d.website_url else "Not set (optional)")

    return sc, fb


# ── CV — 105 pts ─────────────────────────────────────────────
#  Critical: name(8) country(4) desired_role(8) industry(6)
#            employment_type(6) contact_phone(4) contact_email(4) = 40
#  Important: work_history(15) education(12) skills(10)
#             languages(8) bio(10)                                = 55
#  Optional:  photo(4) certifications(3) salary_range(3)
#             governorate(2) willing_to_relocate(2) portfolio_links(0—cv has no portfolio) = 10

def score_cv(d: CVProfile) -> tuple[dict, list[FieldFeedback]]:
    fb, sc = [], {"critical": 0, "important": 0, "optional": 0}

    def add(field, tier, awarded, possible, status, message):
        sc[tier] += awarded
        fb.append(_fb(field, tier, awarded, possible, status, message))

    # ── Critical ──
    nm = d.name.strip()
    add("name","critical",
        8 if len(nm) >= 3 else 0, 8,
        "ok" if len(nm) >= 3 else "fail",
        "Name set" if len(nm) >= 3 else "Name too short")

    add("country","critical",
        4 if d.country.strip() else 0, 4,
        "ok" if d.country.strip() else "fail",
        "Country set" if d.country.strip() else "Country required")

    add("desired_role","critical",
        8 if d.desired_role.strip() else 0, 8,
        "ok" if d.desired_role.strip() else "fail",
        "Desired role set" if d.desired_role.strip() else "Desired role required")

    add("industry","critical",
        6 if d.industry.strip() else 0, 6,
        "ok" if d.industry.strip() else "fail",
        "Industry set" if d.industry.strip() else "Industry required")

    add("employment_type","critical",
        6 if d.employment_type.strip() else 0, 6,
        "ok" if d.employment_type.strip() else "fail",
        "Employment type set" if d.employment_type.strip() else "Employment type required")

    add("contact_phone","critical",
        4 if d.contact_phone else 0, 4,
        "ok" if d.contact_phone else "warning",
        "Phone provided" if d.contact_phone else "Phone missing — employers need this to reach you")

    add("contact_email","critical",
        4 if d.contact_email else 0, 4,
        "ok" if d.contact_email else "warning",
        "Email provided" if d.contact_email else "Email missing")

    # ── Important ──
    n_work = len(d.work_history)
    if n_work == 0:
        add("work_history","important",0,15,"warning","No work history — acceptable for entry level only")
    elif n_work == 1:
        add("work_history","important",8,15,"warning","Only 1 work entry")
    else:
        add("work_history","important",15,15,"ok",f"{n_work} work entries")

    n_edu = len(d.education)
    if n_edu == 0:
        add("education","important",0,12,"warning","No education entries")
    elif n_edu == 1:
        add("education","important",8,12,"warning","1 education entry")
    else:
        add("education","important",12,12,"ok",f"{n_edu} education entries")

    n_skills = len(d.skills)
    if n_skills == 0:
        add("skills","important",0,10,"fail","No skills listed")
    elif n_skills < 3:
        add("skills","important",5,10,"warning",f"Only {n_skills} skill(s) — add more")
    else:
        add("skills","important",10,10,"ok",f"{n_skills} skills listed")

    n_lang = len(d.languages)
    if n_lang == 0:
        add("languages","important",0,8,"fail","No languages listed")
    else:
        add("languages","important",8,8,"ok",f"{n_lang} language(s)")

    bio_words = len((d.bio or "").split())
    if bio_words == 0:
        add("bio","important",0,10,"warning","No bio / summary — optional but improves profile")
    elif bio_words < 30:
        add("bio","important",4,10,"warning",f"Bio only {bio_words} words")
    else:
        add("bio","important",10,10,"ok",f"Bio has {bio_words} words")

    # ── Optional ──
    add("photo_url","optional",
        4 if d.photo_url else 0, 4,
        "ok" if d.photo_url else "ok",
        "Photo provided" if d.photo_url else "No photo (optional)")

    add("certifications","optional",
        3 if d.certifications else 0, 3,
        "ok" if d.certifications else "ok",
        f"{len(d.certifications)} certification(s)" if d.certifications else "None listed (optional)")

    if d.salary_min and d.salary_max:
        add("salary_range","optional",3,3,"ok","Salary range set")
    elif d.salary_min or d.salary_max:
        add("salary_range","optional",1,3,"warning","Only one salary bound set")
    else:
        add("salary_range","optional",0,3,"ok","Salary not set (optional)")

    return sc, fb


# ══════════════════════════════════════════════════════════════
#  DECISION THRESHOLDS
# ══════════════════════════════════════════════════════════════

THRESHOLDS = {
    "freelancer": {"max": 85, "approve": 60, "flag": 40},
    "company":    {"max": 96, "approve": 68, "flag": 45},
    "cv":         {"max": 105,"approve": 75, "flag": 50},  # CV auto-activates at ~90% = 94.5 pts
}

# CV auto-activation threshold is 90% = 94.5 → round to 95
CV_AUTO_ACTIVATE_THRESHOLD = 95   # pts


# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT (shared, profile-type-aware)
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are BKOTT's AI profile review agent. BKOTT is an Arab-world business marketplace (freelancers, companies, job seekers).

You receive a pre-scored profile with field-level feedback from a deterministic engine. Your job is to add QUALITATIVE judgment on top of the numbers, then return a final structured JSON decision.

## Profile types and max scores
- freelancer: 85 pts  → approve ≥60, flag 40-59, reject <40
- company:    96 pts  → approve ≥68, flag 45-67, reject <45
- cv:        105 pts  → approve ≥75, flag 50-74, reject <50
  (CV also auto-activates at ≥90% completion = 95 pts, with NO admin review needed — note this in summary)

## Hard-stop rules (auto_reject_triggers array)
If ANY trigger is present, decision MUST be "reject" — no exceptions, no matter the score.

Triggers that must always cause rejection:
1. Profanity / offensive language (شتايم / الفاظ خارجة) in any free-text field
2. Contact info (email, phone, WhatsApp, Telegram, Instagram etc.) in description/bio/headline
   NOTE: contact_email and contact_phone FIELDS are allowed — only reject if found inside description text
3. Placeholder / test content
4. All portfolio items empty (freelancer only)

## Your qualitative checks (beyond the score)
- Is the profile coherent? Do skills / headline / category match?
- Does the bio/description sound like a real person, not a template?
- Is the portfolio/work history specific and believable?
- Are claims plausible (e.g. 2 years experience but claiming Fortune 500 clients)?
- Is the overall tone professional and marketplace-appropriate?
- For companies: does the description match the selected business type?
- For CVs: does desired role match work history?

## Output
Return ONLY valid JSON — no prose, no markdown fences:
{
  "profile_type": "<freelancer|company|cv>",
  "decision": "<approve|reject|flag_for_human>",
  "score": <int>,
  "max_score": <int>,
  "score_breakdown": {"critical": <int>, "important": <int>, "optional": <int>},
  "field_feedback": [
    {"field":"<name>","tier":"<critical|important|optional>","status":"<ok|warning|fail>","message":"<string>","points_awarded":<int>,"points_possible":<int>}
  ],
  "auto_reject_triggers": ["<string>"],
  "summary": "<1-2 sentence admin summary>",
  "applicant_message": "<Arabic-first friendly message to applicant — null if approved>"
}

You may adjust the pre-computed score by ±8 points based on qualitative judgment. Never exceed the profile's max score."""


def build_prompt(profile_type: str, raw_data: dict, scores: dict, feedback: list[FieldFeedback], triggers: list[str]) -> str:
    return json.dumps({
        "profile_type": profile_type,
        "profile_data": raw_data,
        "pre_computed": {
            "base_score": sum(scores.values()),
            "max_score": THRESHOLDS[profile_type]["max"],
            "score_breakdown": scores,
            "field_feedback": [f.model_dump() for f in feedback],
            "auto_reject_triggers": triggers,
            "thresholds": THRESHOLDS[profile_type],
            "note": "Adjust score ±8 based on qualitative assessment only."
        }
    }, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════
#  ROUTE
# ══════════════════════════════════════════════════════════════

@app.post("/review-profile", response_model=ReviewDecision, summary="Review any profile type")
async def review_profile(submission: ProfileSubmission):
    """
    Universal profile review endpoint.

    Send `profile_type` as one of: `freelancer` | `company` | `cv`
    Send `data` as the full profile object for that type.

    The route will:
    1. Validate the data against the correct schema
    2. Run the deterministic scoring engine for that profile type
    3. Scan free-text fields for profanity / contact info / placeholders
    4. Send everything to Claude for qualitative review
    5. Return a structured ReviewDecision
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    ptype = submission.profile_type
    raw   = submission.data

    # ── 1. Parse into the correct model ──
    try:
        if ptype == ProfileType.freelancer:
            profile = FreelancerProfile(**raw)
            scores, feedback = score_freelancer(profile)
        elif ptype == ProfileType.company:
            profile = CompanyProfile(**raw)
            scores, feedback = score_company(profile)
        elif ptype == ProfileType.cv:
            profile = CVProfile(**raw)
            scores, feedback = score_cv(profile)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown profile_type: {ptype}")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Schema validation failed for '{ptype}': {str(e)}")

    # ── 2. Content guard ──
    triggers = scan_content(ptype, raw)

    # ── 3. Build prompt and call Claude ──
    user_msg = build_prompt(ptype, raw, scores, feedback, triggers)

    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": 2000,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": user_msg}],
    }

    async with httpx.AsyncClient(timeout=40.0) as client:
        try:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key":          ANTHROPIC_API_KEY,
                    "anthropic-version":  "2023-06-01",
                    "content-type":       "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Claude API error: {e.response.text}")
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Claude API timed out")

    raw_text = resp.json()["content"][0]["text"].strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        decision = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Claude returned invalid JSON: {e}\n{raw_text[:400]}")

    max_sc = THRESHOLDS[ptype]["max"]
    decision["score"]     = max(0, min(max_sc, decision.get("score", 0)))
    decision["max_score"] = max_sc

    return ReviewDecision(**decision)


# ══════════════════════════════════════════════════════════════
#  UTILITY ROUTES
# ══════════════════════════════════════════════════════════════

@app.get("/schema/{profile_type}", summary="Get the expected JSON structure for a profile type")
async def get_schema(profile_type: str):
    schemas = {
        "freelancer": FreelancerProfile.schema(),
        "company":    CompanyProfile.schema(),
        "cv":         CVProfile.schema(),
    }
    if profile_type not in schemas:
        raise HTTPException(status_code=404, detail=f"Unknown profile_type '{profile_type}'. Options: freelancer, company, cv")
    return schemas[profile_type]


@app.get("/health")
async def health():
    return {"status": "ok", "model": CLAUDE_MODEL, "profile_types": ["freelancer","company","cv"]}


@app.get("/")
async def root():
    return {
        "service":   "BKOTT Profile Review Agent v2",
        "endpoint":  "POST /review-profile",
        "types":     ["freelancer (85 pts)", "company (96 pts)", "cv (105 pts)"],
        "docs":      "/docs",
    }
