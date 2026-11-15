#!/usr/bin/env python3
"""
پروژه درس داده‌کاوی کارشناسی ارشد: سامانه دوزبانه سنجش چندبعدی شباهت متون خبری (فارسی و انگلیسی)
M.Sc. Data Mining Project — Bilingual (Persian & English) Cross-Source News Similarity Miner
Author: Mohammadreza Jabbari (GitHub: mrza25)
"""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import html
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import re
from typing import Dict, List, Set, Tuple
from urllib.parse import parse_qs
import urllib.request
from bs4 import BeautifulSoup
import numpy as np


@dataclass(frozen=True)
class NewsArticle:
    """ساختار داده استاندارد برای نگهداری یک خبر (فارسی یا انگلیسی) همراه با متادیتا و لینک منبع."""
    article_id: str
    title: str
    source_name: str
    source_url: str
    published_date: str
    language: str
    category: str
    content: str


DEFAULT_ARTICLES: Dict[str, NewsArticle] = {
    "irna_iran_palestine": NewsArticle(
        article_id="irna_iran_palestine",
        title="ایران ۲ - فلسطین صفر؛ جوان‌ها امیدوارمان کردند",
        source_name="خبرگزاری جمهوری اسلامی (ایرنا)",
        source_url="https://www.irna.ir/news/86321699/",
        published_date="۲۳ آبان ۱۴۰۵ (2026-11-14)",
        language="fa",
        category="ورزشی - فوتبال ملی",
        content=(
            "تیم ملی فوتبال ایران در اولین بازی تدارکاتی در تورنمنت بصره برابر فلسطین به پیروزی رسید. "
            "تیم ملی فوتبال ایران از ساعت ۱۸:۳۰ در یک بازی تدارکاتی در ورزشگاه بصره به مصاف تیم ملی فلسطین رفت. "
            "امیر قلعه‌نویی تیم ملی را با توجه به تعویق یک ساله جام ملت‌ها با یک ترکیب جوان‌شده راهی بصره کرده است "
            "تا استارت جوانگرایی در تیم ملی با سرعت بیشتری دنبال شود. "
            "در دقیقه ۱۹ در یک موقعیت جدی، سانتر جعفر سلمانی در دهانه دروازه به امیرحسین حسین‌زاده رسید اما مدافع فلسطین با یک تکل توپ را دفع کرد. "
            "در دقیقه ۲۱ ضربه آریا یوسفی روی پاس سلمانی را دروازه‌بان فلسطین روی خط دروازه دفع کرد. "
            "در دقیقه ۲۵ ضربه سر محمد قربانی روی سانتر محمدجواد حسین‌نژاد به تیر دروازه برخورد کرد تا نیمه نخست با تساوی بدون گل ایران و فلسطین به پایان برسد. "
            "در نیمه دوم در دقیقه ۴۷ ضربه بغل پای سرضرب محمد محبی در دو قدمی دروازه را دروازه‌بان فلسطین به بیرون فرستاد. "
            "در دقیقه ۵۰ امیرحسین حسین‌زاده روی پاس محمد محبی صاحب توپ شد اما شوت او با اختلاف کمی از کنار دروازه به بیرون رفت. "
            "در دقیقه ۶۲ ضربه سر محمدمهدی محبی روی سانتر حسین گودرزی از کنار دروازه به بیرون رفت. "
            "عمید صوافطه در دقیقه ۶۴ با دریافت کارت قرمز از بازی اخراج شد تا فلسطین ۱۰ نفره شود و کار شاگردان ایهاب ابوجزر سخت‌تر شود. "
            "سرانجام در دقیقه ۷۶ قفل دروازه فلسطین باز شد و سردار آزمون روی ارسال حسین گودرزی با یک بغل پا ایران را یک بر صفر پیش انداخت. "
            "فشار ایران ادامه پیدا کرد و در دقیقه ۸۴ صالح حردانی روی ارسال حسین گودرزی از چپ توپ را به تور دروازه فلسطین دوخت تا نتیجه ۲ بر صفر شود. "
            "در نهایت تیم ملی فوتبال ایران توانست با نتیجه ۲ بر صفر فلسطین را شکست دهد."
        ),
    ),
    "mehr_iran_palestine": NewsArticle(
        article_id="mehr_iran_palestine",
        title="کارت قرمز به داد قلعه‌نویی رسید؛ پیروزی سخت ایران برابر فلسطین",
        source_name="خبرگزاری مهر",
        source_url="https://www.mehrnews.com/news/7010871/",
        published_date="۲۳ آبان ۱۴۰۵ (2026-11-14)",
        language="fa",
        category="ورزشی - فوتبال ایران",
        content=(
            "شاگردان امیر قلعه‌نویی از دقیقه ۱۱ با ارسال جعفر سلمانی و شوت دنیس درگاهی نخستین موقعیت جدی خود را ایجاد کردند. "
            "در ادامه همکاری بازیکنان ایران از سمت چپ و ارسال خطرناک سلمانی در دقیقه ۱۹ می‌توانست به گل منجر شود اما مدافع فلسطین با تکلی به‌موقع مانع از گلزنی امیرحسین حسین‌زاده شد. "
            "دو دقیقه بعد نیز شوت آریا یوسفی از داخل محوطه جریمه با واکنش دروازه‌بان فلسطین مهار شد. "
            "مهم‌ترین فرصت ایران در نیمه نخست در دقیقه ۲۵ رقم خورد؛ جایی که ارسال محمدجواد حسین‌نژاد از روی نقطه کرنر با ضربه سر محمد قربانی همراه شد اما توپ پس از برخورد به تیر دروازه از گل شدن بازماند. "
            "در نیمه دوم در دقیقه ۴۷ ارسال سعید عزت‌اللهی به محوطه جریمه با ضربه سر سردار آزمون همراه شد و در ادامه ضربه محمد محبی با واکنش دروازه‌بان فلسطین از روی خط دروازه برگشت داده شد. "
            "سه دقیقه بعد محبی با یک پاس عمقی زیبا حسین‌زاده را در موقعیت گلزنی قرار داد اما ضربه او از کنار دروازه فلسطین به بیرون رفت. "
            "در دقیقه ۶۳ ارسال خطرناک حسین گودرزی از سمت چپ با ضربه سر محمدمهدی محبی همراه شد که توپ به بیرون رفت. "
            "یک دقیقه بعد عمید صوافطه به دلیل ضربه زدن به صورت عزت‌اللهی دومین کارت زرد خود را دریافت کرد تا با اخراج از زمین تیم ملی فلسطین ۱۰ نفره به کار خود ادامه دهد. "
            "سرانجام در دقیقه ۷۷ قفل دروازه فلسطین باز شد؛ پاس کوتاه اسماعیل قلی‌زاده به حسین گودرزی رسید و ارسال دقیق این بازیکن با ضربه محکم بغل پای سردار آزمون همراه شد تا مهاجم تیم ملی گل نخست ایران را به ثمر برساند. "
            "در دقیقه ۸۴ تیم ملی ایران بار دیگر از سمت چپ حمله‌ای خطرناک ترتیب داد و حسین گودرزی توپ را به محوطه جریمه فلسطین فرستاد تا صالح حردانی با جاگیری مناسب و ضربه کف پا گل دوم ایران را به ثمر برساند و پیروزی ۲ بر صفر ایران رقم بخورد."
        ),
    ),
    "isna_iran_palestine": NewsArticle(
        article_id="isna_iran_palestine",
        title="برد ۲ گله شاگردان قلعه‌نویی برابر فلسطین ۱۰ نفره",
        source_name="خبرگزاری دانشجویان ایران (ایسنا)",
        source_url="https://www.isna.ir/news/1405082315926/",
        published_date="۲۳ آبان ۱۴۰۵ (2026-11-14)",
        language="fa",
        category="ورزشی - فوتبال، فوتسال",
        content=(
            "تیم ملی فوتبال ایران در دیدار تدارکاتی تورنمنت بصره موفق شد تیم ملی فلسطین ۱۰ نفره را با دو گل شکست دهد. "
            "تیم ملی ایران در دقیقه ۲۵ صاحب ضربه کرنر شد که محمدجواد حسین‌نژاد آن را به محوطه جریمه فلسطین ارسال کرد و در ادامه ضربه سر محمد قربانی به تیر دروازه برخورد کرد. "
            "در دقیقه ۵۰ محمد محبی با یک پاس امیرحسین حسین‌زاده را صاحب موقعیت کرد که شوت او با اختلاف کمی از کنار دروازه به بیرون رفت. "
            "در ادامه در دقیقه ۶۳ حسین گودرزی اقدام به سانتر کرد که ضربه سر محمدمهدی محبی از کنار دروازه به بیرون رفت. "
            "در دقیقه ۶۵ بازیکن فلسطین به دلیل ضربه به صورت سعید عزت‌اللهی از زمین بازی اخراج شد تا فلسطین با ۱۰ بازیکن به کار خود ادامه دهد. "
            "گل نخست ایران در دقیقه ۷۶ به ثمر رسید؛ حسین گودرزی توپ را به محوطه جریمه فلسطین ارسال کرد و سردار آزمون آن را به تور دروازه حریف چسباند. "
            "گل دوم ایران در دقیقه ۸۴ ثبت شد؛ بار دیگر ارسال حسین گودرزی به محوطه جریمه فلسطین انجام شد و این بار صالح حردانی موفق به گلزنی شد. "
            "شاگردان امیر قلعه‌نویی سه‌شنبه ۲۶ آبان در بازی بعدی به مصاف عراق خواهند رفت."
        ),
    ),
    "football360_iran_palestine": NewsArticle(
        article_id="football360_iran_palestine",
        title="ایران ۲-۰ فلسطین؛ گودرزی ساخت، باتجربه‌ها گل کردند",
        source_name="فوتبال ۳۶۰ (Football360)",
        source_url="https://football360.ir/post/20261114127/",
        published_date="۲۳ آبان ۱۴۰۵ (2026-11-14)",
        language="fa",
        category="ورزشی - فوتبال ملی",
        content=(
            "تیم ملی ایران در نخستین مسابقه تورنمنت بصره با دو گل سردار آزمون و صالح حردانی از سد فلسطین گذشت. "
            "در دقیقه ۶۴ شرایط برای فلسطین دشوارتر شد و عمید صوافطه مدافع شماره ۵ این تیم پس از برخورد با صورت سعید عزت‌اللهی با دریافت کارت زرد دوم از زمین مسابقه اخراج شد و فلسطین مجبور شد ادامه بازی را با یک بازیکن کمتر دنبال کند. "
            "سرانجام در دقیقه ۷۶ برتری عددی و فشار مداوم ایران نتیجه داد؛ اسماعیل قلی‌زاده و حسین گودرزی در سمت چپ زمین با یکدیگر همکاری کردند و گودرزی با یک ارسال بلند سردار آزمون را صاحب موقعیت کرد و کاپیتان تیم ملی با ضربه‌ای دقیق دروازه فلسطین را باز کرد. "
            "ایران که پس از گل نخست همچنان به حملاتش ادامه می‌داد در دقیقه ۸۳ به گل دوم رسید؛ این بار ارسال حسین گودرزی با حضور به‌موقع صالح حردانی در محوطه جریمه همراه شد و مدافع استقلالی تیم ملی با قطع توپ در شش‌قدم اختلاف را به دو گل رساند. "
            "در نهایت تیم ملی ایران با پیروزی ۲ بر صفر مقابل فلسطین نخستین گام خود در تورنمنت بصره را محکم برداشت و شاگردان قلعه‌نویی روز سه‌شنبه ۲۶ آبان مقابل عراق قرار می‌گیرند."
        ),
    ),
    "irna_ghoddos_retirement": NewsArticle(
        article_id="irna_ghoddos_retirement",
        title="تصمیم عجیب ستاره تیم ملی فوتبال؛ «سامان» با یوزها خداحافظی کرد!",
        source_name="خبرگزاری جمهوری اسلامی (ایرنا)",
        source_url="https://www.irna.ir/news/86317660/",
        published_date="۱۹ آبان ۱۴۰۵ (2026-11-09)",
        language="fa",
        category="ورزشی - فوتبال ملی",
        content=(
            "ستاره تیم ملی فوتبال کشورمان با انتشار پستی در صفحه مجازی خود به صورت رسمی از بازی‌های ملی خداحافظی کرد. "
            "تیم ملی فوتبال کشورمان که خود را برای حضور در تورنمنت عراق در فیفادی آبان ماه آماده می‌کند با دعوت از ۲۹ بازیکن از طرف امیر قلعه‌نویی تمرینات خود را در تهران آغاز کرد. "
            "سامان قدوس ستاره ۳۳ ساله تیم ملی فوتبال کشورمان در لیست تیم ملی حضور نداشت و تصمیم گرفت با انتشار پیامی احساسی به صورت رسمی از بازی‌های ملی خداحافظی کند. "
            "سامان قدوس طی ۷۰ بازی ملی موفق به زدن ۲ گل شد و در ۳ دوره جام جهانی و ۲ جام ملت‌های آسیا برای تیم ملی ایران به میدان رفت."
        ),
    ),
    "guardian_iran_nz_wc2026": NewsArticle(
        article_id="guardian_iran_nz_wc2026",
        title="Iran twice come from behind to claim draw in World Cup opener with New Zealand",
        source_name="The Guardian",
        source_url="https://www.theguardian.com/football/2026/jun/16/iran-new-zealand-world-cup-2026-group-g-match-report",
        published_date="2026-06-16",
        language="en",
        category="Sports - World Cup 2026",
        content=(
            "Iran twice came from behind to claim a thrilling 2-2 draw against New Zealand in their World Cup 2026 Group G opener in Los Angeles. "
            "New Zealand twice took the lead through forward Elijah Just in the 7th and 54th minutes, with Nottingham Forest striker Chris Wood providing the assist on both goals past goalkeeper Alireza Beiranvand. "
            "Iran captain Mehdi Taremi struck the post after surging to the edge of the penalty area from inside his own half. "
            "Iran levelled in the 32nd minute when right-back Ramin Rezaeian poked the ball home from close range after Saman Ghoddos and Shahriar Moghanloo combined inside the box. "
            "After Elijah Just restored the lead for the All Whites in the 54th minute following another pass from Chris Wood, Iran equalised again in the 64th minute when Mohammad Mohebbi guided a header in off the post from Ramin Rezaeian's cross. "
            "The draw leaves all four teams in Group G level on one point after Belgium and Egypt also drew 1-1 earlier on Monday."
        ),
    ),
    "arabnews_reuters_iran_nz_wc2026": NewsArticle(
        article_id="arabnews_reuters_iran_nz_wc2026",
        title="Iran fight back to draw 2-2 with New Zealand at World Cup",
        source_name="Arab News (Reuters)",
        source_url="https://www.arabnews.com/football/iran-fight-back-to-draw-2-2-with-new-zealand-at-world-cup-2647332",
        published_date="2026-06-16",
        language="en",
        category="Sports - Football World Cup",
        content=(
            "Iran twice came from behind to earn a 2-2 draw with New Zealand in their opening World Cup match on Monday in Los Angeles after an end-to-end Group G contest. "
            "New Zealand went ahead in the 7th minute when Elijah Just volleyed home from inside the box after being set up by striker Chris Wood. "
            "Iran deservedly levelled in the 32nd minute, with Ramin Rezaeian stabbing the ball in from close range to send the crowd at the Los Angeles stadium into raptures. "
            "Motherwell winger Elijah Just restored New Zealand's lead in the 54th minute, lifting the ball over goalkeeper Alireza Beiranvand after another assist from captain Chris Wood. "
            "However, Iran hit back again 10 minutes later in the 64th minute when Mohammad Mohebbi glanced a header in off the back post from Ramin Rezaeian's cross to equalize for a 2-2 draw. "
            "The result means all teams in Group G have one point after Belgium drew 1-1 with Egypt earlier on Monday, and Iran will next face Belgium in Los Angeles."
        ),
    ),
}


def get_default_pair() -> Tuple[NewsArticle, NewsArticle]:
    """برگرداندن جفت خبر پیش‌فرض فارسی (گزارش بازی ایران و فلسطین از ایرنا و خبرگزاری مهر)."""
    return DEFAULT_ARTICLES["irna_iran_palestine"], DEFAULT_ARTICLES["mehr_iran_palestine"]


def get_default_english_pair() -> Tuple[NewsArticle, NewsArticle]:
    """برگرداندن جفت خبر پیش‌فرض انگلیسی (گزارش بازی ایران و نیوزیلند در جام جهانی ۲۰۲۶ از گاردین و رویترز)."""
    return DEFAULT_ARTICLES["guardian_iran_nz_wc2026"], DEFAULT_ARTICLES["arabnews_reuters_iran_nz_wc2026"]


PERSIAN_STOPWORDS: Set[str] = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن", "یک", "شد", "تا",
    "خود", "بر", "هم", "نیز", "اما", "کرد", "شود", "بود", "دارد", "شدند", "های", "ها",
    "می", "بر", "روی", "پس", "باید", "کند", "کنند", "کردند", "بودند", "داشته", "باشد",
    "باشند", "سوی", "بدون", "چند", "هر", "یا", "اگر", "همه", "دیگر", "تنها", "همچنان",
    "رسید", "رفت", "داد", "گرفت", "شمار", "ادامه", "جای", "همراه", "کنار", "توسط",
    "تیم", "بازی", "برگزار", "قرار", "توانست", "موفق", "نهایت", "ابتدا", "سپس", "حالی",
    "شرایطی", "صورت", "طریق", "عنوان", "درباره", "برابر", "مقابل", "طبق", "گزارش",
}

ENGLISH_STOPWORDS: Set[str] = {
    "the", "and", "in", "to", "of", "a", "an", "for", "on", "with", "as", "by", "at",
    "from", "was", "were", "is", "are", "be", "been", "being", "that", "this", "it",
    "its", "their", "his", "her", "they", "them", "he", "she", "we", "you", "who",
    "whom", "which", "what", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "after", "before", "into", "over", "under", "again", "further",
    "then", "once", "here", "there", "up", "down", "out", "off", "above", "below",
    "between", "through", "during", "against", "about", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "might", "must", "shall", "however",
    "while", "also", "but", "or", "if", "because", "until"
}

ALL_STOPWORDS: Set[str] = PERSIAN_STOPWORDS | ENGLISH_STOPWORDS

DIGIT_TRANSLATION_TABLE = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

CHAR_NORMALIZATION_TABLE = str.maketrans({
    "ي": "ی", "ى": "ی", "ئ": "ی", "ك": "ک", "ؤ": "و",
    "إ": "ا", "أ": "ا", "ٱ": "ا", "ة": "ه", "ۀ": "ه", "\u200c": " ",
})

ORTHOGRAPHIC_PATTERNS: List[Tuple[str, str]] = [
    (r"(?<!\S)دو(?!\S)", "2"),
    (r"(?<!\S)سه(?!\S)", "3"),
    (r"(?<!\S)چهار(?!\S)", "4"),
    (r"(?<!\S)پنج(?!\S)", "5"),
    (r"(?<!\S)ده(?!\S)", "10"),
    (r"(?<!\S)یازده(?!\S)", "11"),
    (r"(?<!\S)صفر(?!\S)", "0"),
    (r"(?<!\S)نخستین(?!\S)", "اولین"),
    (r"(?<!\S)نخست(?!\S)", "اول"),
    (r"(?<!\S)دربی(?!\S)", "شهرآورد"),
    (r"(?<!\S)پایتخت(?!\S)", "تهران"),
    (r"(?<!\S)گلزنی(?!\S)", "گل"),
    (r"(?<!\S)تدارکاتی(?!\S)", "دوستانه"),
    (r"\bseventh\b", "7th"),
    (r"\bfirst\b", "1st"),
    (r"\bsecond\b", "2nd"),
    (r"\bthird\b", "3rd"),
    (r"\btwo\b", "2"),
    (r"\bthree\b", "3"),
    (r"\bfour\b", "4"),
    (r"\bten\b", "10"),
    (r"\bnil\b", "0"),
    (r"\blevelled\b", "equalized"),
    (r"\bleveled\b", "equalized"),
    (r"\bequalised\b", "equalized"),
    (r"\bequaliser\b", "equalizer"),
    (r"\bdefence\b", "defense"),
    (r"\bcentre\b", "center"),
]

PROTECTED_WORDS: Set[str] = {
    "ایران", "فلسطین", "تهران", "جوان", "کاپیتان", "میدان", "سامان", "آبان",
    "قلعه", "نویی", "گودرزی", "حردانی", "قربانی", "سلمانی", "یوسفی", "محبی",
    "طارمی", "عزت", "اللهی", "تورنمنت", "فوتبال", "استقلال", "پرسپولیس", "بیرون",
    "iran", "zealand", "taremi", "rezaeian", "mohebbi", "beiranvand", "ghoddos",
    "elijah", "chris", "wood", "belgium", "egypt", "angeles", "monday", "sunday",
    "morning", "evening", "stadium", "captain", "winger", "striker", "header", "corner"
}

ENGLISH_IRREGULAR_LEMMAS: Dict[str, str] = {
    "came": "come", "went": "go", "took": "take", "taken": "take", "made": "make",
    "struck": "strike", "led": "lead", "won": "win", "lost": "lose", "drew": "draw",
    "drawn": "draw", "hit": "hit", "set": "set", "men": "man", "feet": "foot",
}

PERSIAN_SUFFIXES: Tuple[str, ...] = (
    "هایشان", "هایمان", "هایتان", "هایی", "های", "ها",
    "شان", "مان", "تان", "اش", "ام", "ات", "ترین", "تر", "گان", "ان", "ش"
)

PERSIAN_PREFIXES: Tuple[str, ...] = ("می‌", "نمی‌", "بی‌", "می ", "نمی ", "بی ")


@dataclass
class PreprocessedDocument:
    """خروجی کامل مرحله پیش‌پردازش یک سند خبری (فارسی یا انگلیسی)."""
    language: str
    raw_text: str
    normalized_text: str
    sentences: List[str]
    raw_tokens: List[str]
    filtered_tokens: List[str]
    stemmed_tokens: List[str]
    word_ngrams: List[str]
    char_shingles: Set[str]


class PersianNewsPreprocessor:
    """پایپ‌لاین پیش‌پردازش دوزبانه (فارسی و انگلیسی) ویژه پروژه‌های داده‌کاوی و بازیابی اطلاعات."""

    def __init__(
        self,
        remove_stopwords: bool = True,
        apply_stemming: bool = True,
        ngram_range: Tuple[int, int] = (1, 2),
        char_shingle_k: int = 4,
    ) -> None:
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming
        self.ngram_range = ngram_range
        self.char_shingle_k = char_shingle_k

    @staticmethod
    def detect_language(text: str) -> str:
        persian_chars = len(re.findall(r"[آ-ی]", text))
        latin_chars = len(re.findall(r"[a-zA-Z]", text))
        if persian_chars == 0 and latin_chars == 0:
            return "unknown"
        if latin_chars > persian_chars * 1.5:
            return "en"
        if persian_chars > latin_chars * 1.5:
            return "fa"
        return "mixed"

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        cleaned = re.sub(r"<[^>]+>", " ", text)
        cleaned = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", cleaned)
        cleaned = cleaned.translate(CHAR_NORMALIZATION_TABLE)
        cleaned = cleaned.translate(DIGIT_TRANSLATION_TABLE)
        cleaned = re.sub(r"['’]s\b", "", cleaned)
        cleaned = cleaned.lower()
        for pattern, replacement in ORTHOGRAPHIC_PATTERNS:
            cleaned = re.sub(pattern, replacement, cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"\s+", " ", text).strip()
        raw_parts = re.split(r"[.!؟?;؛\n]+", cleaned)
        return [part.strip() for part in raw_parts if len(part.strip()) > 10]

    @staticmethod
    def tokenize(normalized_text: str) -> List[str]:
        if not normalized_text:
            return []
        tokens = re.findall(r"[آ-یa-zA-Z0-9]+", normalized_text.lower())
        return [t for t in tokens if len(t) > 1 or t.isdigit()]

    @staticmethod
    def stem_english_word(word: str) -> str:
        if len(word) <= 3 or word.isdigit() or word in PROTECTED_WORDS:
            return word
        if word in ENGLISH_IRREGULAR_LEMMAS:
            return ENGLISH_IRREGULAR_LEMMAS[word]

        w = word
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("ing") and len(w) >= 6:
            base = w[:-3]
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
        if w.endswith("ed") and len(w) >= 5:
            base = w[:-2]
            if base.endswith("i"):
                return base[:-1] + "y"
            if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
        if w.endswith("es") and len(w) > 4 and not w.endswith(("aes", "ees", "oes")):
            if w.endswith(("ches", "shes", "xes", "zes", "sses")):
                return w[:-2]
            return w[:-1]
        if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
            return w[:-1]
        if w.endswith("ly") and len(w) > 5:
            return w[:-2]
        return w

    @classmethod
    def stem_word(cls, word: str) -> str:
        cleaned = word.replace("\u200c", "").strip().lower()
        if len(cleaned) <= 3 or cleaned.isdigit() or cleaned in PROTECTED_WORDS:
            return cleaned

        if re.match(r"^[a-z0-9]+$", cleaned):
            return cls.stem_english_word(cleaned)

        stemmed = cleaned
        for prefix in PERSIAN_PREFIXES:
            p_clean = prefix.replace("\u200c", "").strip()
            if stemmed.startswith(p_clean) and len(stemmed) - len(p_clean) >= 3:
                stemmed = stemmed[len(p_clean):]
                break

        if stemmed in PROTECTED_WORDS:
            return stemmed

        for suffix in PERSIAN_SUFFIXES:
            if stemmed.endswith(suffix) and len(stemmed) - len(suffix) >= 3:
                return stemmed[:-len(suffix)]

        return stemmed

    @staticmethod
    def generate_word_ngrams(tokens: List[str], ngram_range: Tuple[int, int] = (1, 2)) -> List[str]:
        min_n, max_n = ngram_range
        ngrams: List[str] = []
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(n_tokens - n + 1):
                ngrams.append("_".join(tokens[i : i + n]))
        return ngrams

    @staticmethod
    def generate_char_shingles(text: str, k: int = 4) -> Set[str]:
        compact = re.sub(r"[^آ-یa-zA-Z0-9 ]", "", text.lower())
        compact = re.sub(r"\s+", " ", compact).strip()
        if len(compact) < k:
            return {compact} if compact else set()
        return {compact[i : i + k] for i in range(len(compact) - k + 1)}

    def process(self, text: str) -> PreprocessedDocument:
        lang = self.detect_language(text)
        normalized = self.normalize(text)
        sentences = self.split_sentences(text)
        raw_tokens = self.tokenize(normalized)

        filtered = [t for t in raw_tokens if t not in ALL_STOPWORDS] if self.remove_stopwords else list(raw_tokens)
        if self.apply_stemming:
            stemmed = [self.stem_word(t) for t in filtered]
            if self.remove_stopwords:
                stemmed = [t for t in stemmed if t not in ALL_STOPWORDS]
        else:
            stemmed = list(filtered)

        word_ngrams = self.generate_word_ngrams(stemmed, self.ngram_range)
        shingles = self.generate_char_shingles(" ".join(stemmed), k=self.char_shingle_k)

        return PreprocessedDocument(
            language=lang,
            raw_text=text,
            normalized_text=normalized,
            sentences=sentences,
            raw_tokens=raw_tokens,
            filtered_tokens=filtered,
            stemmed_tokens=stemmed,
            word_ngrams=word_ngrams,
            char_shingles=shingles,
        )


@dataclass
class SentenceAlignmentMatch:
    sentence_1_index: int
    sentence_2_index: int
    sentence_1_text: str
    sentence_2_text: str
    similarity_score: float
    shared_terms: List[str]


@dataclass
class SimilarityReport:
    tfidf_cosine_similarity: float
    unigram_cosine_similarity: float
    jaccard_word_similarity: float
    dice_word_similarity: float
    overlap_coefficient: float
    char_shingle_jaccard: float
    minhash_estimated_jaccard: float
    lsh_candidate_match: bool
    euclidean_distance: float
    sentence_alignment_score: float
    hybrid_overall_score: float
    similarity_verdict: str
    doc1_token_count: int
    doc2_token_count: int
    doc1_unique_terms: int
    doc2_unique_terms: int
    shared_unique_terms_count: int
    top_shared_keywords: List[Tuple[str, float, float, float]]
    doc1_exclusive_keywords: List[Tuple[str, float]]
    doc2_exclusive_keywords: List[Tuple[str, float]]
    sentence_similarity_matrix: np.ndarray
    top_aligned_sentences: List[SentenceAlignmentMatch]
    doc1_sentences: List[str]
    doc2_sentences: List[str]


class TFIDFVectorizerScratch:
    """بردارساز Sublinear TF-IDF با IDF هموارشده و نرمال‌سازی L2."""

    def __init__(self, sublinear_tf: bool = True) -> None:
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_: np.ndarray = np.array([])

    def fit_transform(self, tokenized_docs: List[List[str]]) -> np.ndarray:
        n_docs = len(tokenized_docs)
        vocab_set: Set[str] = set()
        for doc in tokenized_docs:
            vocab_set.update(doc)

        self.feature_names_ = sorted(vocab_set)
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}
        n_features = len(self.feature_names_)
        if n_features == 0:
            return np.zeros((n_docs, 1), dtype=float)

        df = np.zeros(n_features, dtype=float)
        tf_matrix = np.zeros((n_docs, n_features), dtype=float)

        for doc_idx, doc_tokens in enumerate(tokenized_docs):
            counts = Counter(doc_tokens)
            for term, count in counts.items():
                feat_idx = self.vocabulary_[term]
                df[feat_idx] += 1.0
                tf_matrix[doc_idx, feat_idx] = (1.0 + math.log(count)) if self.sublinear_tf else float(count)

        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        tfidf = tf_matrix * self.idf_
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return tfidf / norms


class MinHashLSH:
    """الگوریتم MinHash و Locality-Sensitive Hashing (LSH) با ۱۲۸ جایگشت و عدد اول مرسن."""

    def __init__(self, num_perm: int = 128, bands: int = 32, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self.mersenne_prime = (1 << 61) - 1
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, self.mersenne_prime - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, self.mersenne_prime - 1, size=num_perm, dtype=np.uint64)

    def compute_signature(self, shingles: Set[str]) -> np.ndarray:
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint64)
        shingle_hashes = np.array(
            [int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16) for s in shingles],
            dtype=np.uint64,
        )
        p = np.uint64(self.mersenne_prime)
        hashes = (np.outer(shingle_hashes, self.a) + self.b) % p
        return np.min(hashes, axis=0)

    def estimate_jaccard(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        return float(np.mean(sig1 == sig2))

    def is_lsh_candidate(self, sig1: np.ndarray, sig2: np.ndarray) -> bool:
        for band_idx in range(self.bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            if np.array_equal(sig1[start:end], sig2[start:end]):
                return True
        return False


class NewsSimilarityEngine:
    """موتور دوزبانه سنجش شباهت متون خبری (فارسی و انگلیسی)."""

    def __init__(self, preprocessor: PersianNewsPreprocessor | None = None) -> None:
        self.preprocessor = preprocessor or PersianNewsPreprocessor()
        self.minhash = MinHashLSH(num_perm=128, bands=32, seed=42)

    @staticmethod
    def cosine_similarity_vec(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = float(np.linalg.norm(v1))
        norm2 = float(np.linalg.norm(v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.clip(np.dot(v1, v2) / (norm1 * norm2), 0.0, 1.0))

    @staticmethod
    def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
        if not set1 and not set2:
            return 1.0
        union_size = len(set1 | set2)
        return len(set1 & set2) / union_size if union_size > 0 else 0.0

    @staticmethod
    def dice_similarity(set1: Set[str], set2: Set[str]) -> float:
        total = len(set1) + len(set2)
        return (2.0 * len(set1 & set2)) / total if total > 0 else 0.0

    @staticmethod
    def overlap_coefficient(set1: Set[str], set2: Set[str]) -> float:
        min_len = min(len(set1), len(set2))
        return len(set1 & set2) / min_len if min_len > 0 else 0.0

    def _compute_sentence_alignment(
        self, doc1: PreprocessedDocument, doc2: PreprocessedDocument
    ) -> Tuple[np.ndarray, float, List[SentenceAlignmentMatch]]:
        sents1 = doc1.sentences or [doc1.normalized_text]
        sents2 = doc2.sentences or [doc2.normalized_text]
        proc_sents1 = [self.preprocessor.process(s).stemmed_tokens for s in sents1]
        proc_sents2 = [self.preprocessor.process(s).stemmed_tokens for s in sents2]

        vec = TFIDFVectorizerScratch(sublinear_tf=True)
        tfidf_all = vec.fit_transform(proc_sents1 + proc_sents2)
        n1 = len(sents1)
        sim_matrix = np.clip(np.dot(tfidf_all[:n1], tfidf_all[n1:].T), 0.0, 1.0)

        best_1_to_2 = float(np.mean(np.max(sim_matrix, axis=1))) if sim_matrix.size > 0 else 0.0
        best_2_to_1 = float(np.mean(np.max(sim_matrix, axis=0))) if sim_matrix.size > 0 else 0.0
        alignment_score = 0.5 * (best_1_to_2 + best_2_to_1)

        matches: List[SentenceAlignmentMatch] = []
        for i in range(len(sents1)):
            j = int(np.argmax(sim_matrix[i]))
            matches.append(
                SentenceAlignmentMatch(
                    sentence_1_index=i + 1,
                    sentence_2_index=j + 1,
                    sentence_1_text=sents1[i],
                    sentence_2_text=sents2[j],
                    similarity_score=float(sim_matrix[i, j]),
                    shared_terms=sorted(set(proc_sents1[i]) & set(proc_sents2[j])),
                )
            )
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return sim_matrix, alignment_score, matches

    @staticmethod
    def _interpret_score(score: float) -> str:
        if score >= 0.65:
            return "بسیار بالا (پوشش یک رویداد خبری واحد با جزئیات مشترک فراوان)"
        if score >= 0.45:
            return "بالا (پوشش یک رویداد یکسان با زاویه دید یا نگارش متفاوت)"
        if score >= 0.25:
            return "متوسط (موضوع کلی مشترک در یک حوزه ورزشی، اما رویداد یا جزئیات متفاوت)"
        return "پایین (دو خبر مجزا با اشتراک معنایی اندک)"

    def compare(self, text1: str, text2: str) -> SimilarityReport:
        doc1 = self.preprocessor.process(text1)
        doc2 = self.preprocessor.process(text2)

        ngram_vec = TFIDFVectorizerScratch(sublinear_tf=True)
        ngram_tfidf = ngram_vec.fit_transform([doc1.word_ngrams, doc2.word_ngrams])
        tfidf_cosine = self.cosine_similarity_vec(ngram_tfidf[0], ngram_tfidf[1])
        euclidean_dist = float(np.linalg.norm(ngram_tfidf[0] - ngram_tfidf[1]))

        uni_vec = TFIDFVectorizerScratch(sublinear_tf=True)
        uni_tfidf = uni_vec.fit_transform([doc1.stemmed_tokens, doc2.stemmed_tokens])
        unigram_cosine = self.cosine_similarity_vec(uni_tfidf[0], uni_tfidf[1])

        set1, set2 = set(doc1.stemmed_tokens), set(doc2.stemmed_tokens)
        jaccard_word = self.jaccard_similarity(set1, set2)
        dice_word = self.dice_similarity(set1, set2)
        overlap_coeff = self.overlap_coefficient(set1, set2)

        char_jaccard = self.jaccard_similarity(doc1.char_shingles, doc2.char_shingles)
        sig1 = self.minhash.compute_signature(doc1.char_shingles)
        sig2 = self.minhash.compute_signature(doc2.char_shingles)
        minhash_est = self.minhash.estimate_jaccard(sig1, sig2)
        lsh_cand = self.minhash.is_lsh_candidate(sig1, sig2)

        sim_matrix, alignment_score, top_matches = self._compute_sentence_alignment(doc1, doc2)
        hybrid_score = 0.35 * unigram_cosine + 0.25 * dice_word + 0.20 * overlap_coeff + 0.20 * alignment_score

        shared_kw: List[Tuple[str, float, float, float]] = []
        doc1_excl: List[Tuple[str, float]] = []
        doc2_excl: List[Tuple[str, float]] = []
        for idx, term in enumerate(uni_vec.feature_names_):
            w1, w2 = float(uni_tfidf[0][idx]), float(uni_tfidf[1][idx])
            if w1 > 0 and w2 > 0:
                shared_kw.append((term, w1, w2, w1 * w2))
            elif w1 > 0:
                doc1_excl.append((term, w1))
            elif w2 > 0:
                doc2_excl.append((term, w2))

        shared_kw.sort(key=lambda x: x[3], reverse=True)
        doc1_excl.sort(key=lambda x: x[1], reverse=True)
        doc2_excl.sort(key=lambda x: x[1], reverse=True)

        return SimilarityReport(
            tfidf_cosine_similarity=tfidf_cosine,
            unigram_cosine_similarity=unigram_cosine,
            jaccard_word_similarity=jaccard_word,
            dice_word_similarity=dice_word,
            overlap_coefficient=overlap_coeff,
            char_shingle_jaccard=char_jaccard,
            minhash_estimated_jaccard=minhash_est,
            lsh_candidate_match=lsh_cand,
            euclidean_distance=euclidean_dist,
            sentence_alignment_score=alignment_score,
            hybrid_overall_score=hybrid_score,
            similarity_verdict=self._interpret_score(hybrid_score),
            doc1_token_count=len(doc1.stemmed_tokens),
            doc2_token_count=len(doc2.stemmed_tokens),
            doc1_unique_terms=len(set1),
            doc2_unique_terms=len(set2),
            shared_unique_terms_count=len(set1 & set2),
            top_shared_keywords=shared_kw[:15],
            doc1_exclusive_keywords=doc1_excl[:10],
            doc2_exclusive_keywords=doc2_excl[:10],
            sentence_similarity_matrix=sim_matrix,
            top_aligned_sentences=top_matches[:6],
            doc1_sentences=doc1.sentences,
            doc2_sentences=doc2.sentences,
        )


def fetch_news_from_url(url: str, timeout: int = 12) -> str:
    """وب‌اسکریپر ۴ لایه برای استخراج تمیز متن خبر از سایت‌های خبری انگلیسی و فارسی."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,fa;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        html_bytes = resp.read()
    soup = BeautifulSoup(html_bytes, "html.parser")

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "{}")
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict) and "articleBody" in item and len(str(item["articleBody"])) > 120:
                    return str(item["articleBody"]).strip()
        except Exception:
            continue

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form"]):
        tag.decompose()

    container = soup.find("article") or soup.find("main") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    long_paragraphs = [p for p in paragraphs if len(p) > 35]
    if long_paragraphs:
        return " ".join(long_paragraphs)

    meta_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        return str(meta_desc["content"]).strip()

    return soup.get_text(" ", strip=True)


def format_console_report(title1: str, source1: str, title2: str, source2: str, report: SimilarityReport) -> str:
    lines = [
        "=" * 78,
        "  سامانه داده‌کاوی و سنجش شباهت متون خبری (Bilingual News Similarity Miner)",
        "=" * 78,
        f"خبر اول  [{source1}]: {title1}",
        f"خبر دوم  [{source2}]: {title2}",
        "-" * 78,
        f"   * شباهت کسینوسی تک‌واژه (Unigram TF-IDF Cosine)   : {report.unigram_cosine_similarity:.4f} ({report.unigram_cosine_similarity*100:.1f}%)",
        f"   * شباهت کسینوسی N-gram (1-2 gram TF-IDF Cosine)  : {report.tfidf_cosine_similarity:.4f} ({report.tfidf_cosine_similarity*100:.1f}%)",
        f"   * ضریب شباهت سورن‌سن-دایس (Sørensen–Dice)        : {report.dice_word_similarity:.4f} ({report.dice_word_similarity*100:.1f}%)",
        f"   * ضریب شباهت جاکارد واژگانی (Word Jaccard)       : {report.jaccard_word_similarity:.4f} ({report.jaccard_word_similarity*100:.1f}%)",
        f"   * ضریب هم‌پوشانی (Szymkiewicz–Simpson Overlap)   : {report.overlap_coefficient:.4f} ({report.overlap_coefficient*100:.1f}%)",
        f"   * تخمین امضای MinHash (128 Permutations)         : {report.minhash_estimated_jaccard:.4f} ({report.minhash_estimated_jaccard*100:.1f}%)",
        f"   * کاندیدای مشابه در LSH (32 Bands × 4 Rows)      : {'بله (True)' if report.lsh_candidate_match else 'خیر (False)'}",
        f"   * امتیاز هم‌ترازی جملات (Sentence Alignment)     : {report.sentence_alignment_score:.4f} ({report.sentence_alignment_score*100:.1f}%)",
        f"   >>> شاخص ترکیبی نهایی شباهت (Hybrid Score)       : {report.hybrid_overall_score:.4f} ({report.hybrid_overall_score*100:.1f}%)",
        f"   >>> ارزیابی نهایی مدل: {report.similarity_verdict}",
        "=" * 78,
    ]
    return "\n".join(lines)


def resolve_inputs(args: argparse.Namespace) -> Tuple[str, str, str, str, str, str]:
    if args.text1 and args.text2:
        return (args.title1 or "Custom News 1", args.source1 or "Source 1", args.text1, args.title2 or "Custom News 2", args.source2 or "Source 2", args.text2)
    if args.file1 and args.file2:
        with open(args.file1, "r", encoding="utf-8") as f1, open(args.file2, "r", encoding="utf-8") as f2:
            return (args.title1 or args.file1, args.source1 or "File 1", f1.read(), args.title2 or args.file2, args.source2 or "File 2", f2.read())
    if args.url1 and args.url2:
        return (args.title1 or "Web News 1", args.source1 or args.url1, fetch_news_from_url(args.url1), args.title2 or "Web News 2", args.source2 or args.url2, fetch_news_from_url(args.url2))
    if args.preset == "isna_vs_football360":
        a1, a2 = DEFAULT_ARTICLES["isna_iran_palestine"], DEFAULT_ARTICLES["football360_iran_palestine"]
        return a1.title, a1.source_name, a1.content, a2.title, a2.source_name, a2.content
    if args.preset == "english_guardian_vs_reuters":
        a1, a2 = get_default_english_pair()
        return a1.title, a1.source_name, a1.content, a2.title, a2.source_name, a2.content
    if args.preset == "different_football_news":
        a1, a2 = DEFAULT_ARTICLES["irna_iran_palestine"], DEFAULT_ARTICLES["irna_ghoddos_retirement"]
        return a1.title, a1.source_name, a1.content, a2.title, a2.source_name, a2.content
    a1, a2 = get_default_pair()
    return a1.title, a1.source_name, a1.content, a2.title, a2.source_name, a2.content


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilingual Persian & English News Similarity Miner")
    parser.add_argument("--text1", type=str)
    parser.add_argument("--text2", type=str)
    parser.add_argument("--file1", type=str)
    parser.add_argument("--file2", type=str)
    parser.add_argument("--url1", type=str)
    parser.add_argument("--url2", type=str)
    parser.add_argument("--title1", type=str)
    parser.add_argument("--title2", type=str)
    parser.add_argument("--source1", type=str)
    parser.add_argument("--source2", type=str)
    parser.add_argument("--preset", choices=["irna_vs_mehr", "isna_vs_football360", "english_guardian_vs_reuters", "different_football_news"], default="irna_vs_mehr")
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()

    title1, source1, text1, title2, source2, text2 = resolve_inputs(args)
    engine = NewsSimilarityEngine()
    report = engine.compare(text1, text2)
    print(format_console_report(title1, source1, title2, source2, report))

    if args.json_out:
        payload = {
            "news_1": {"title": title1, "source": source1},
            "news_2": {"title": title2, "source": source2},
            "metrics": {
                "unigram_cosine_similarity": round(report.unigram_cosine_similarity, 4),
                "tfidf_cosine_similarity": round(report.tfidf_cosine_similarity, 4),
                "dice_word_similarity": round(report.dice_word_similarity, 4),
                "jaccard_word_similarity": round(report.jaccard_word_similarity, 4),
                "overlap_coefficient": round(report.overlap_coefficient, 4),
                "minhash_estimated_jaccard": round(report.minhash_estimated_jaccard, 4),
                "lsh_candidate_match": report.lsh_candidate_match,
                "sentence_alignment_score": round(report.sentence_alignment_score, 4),
                "hybrid_overall_score": round(report.hybrid_overall_score, 4),
                "verdict": report.similarity_verdict,
            },
        }
        with open(args.json_out, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
