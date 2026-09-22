"""
Synthetic review data generator for smoke testing.

Generates ~300 deterministic reviews across 6 categories:
Electronics, Beauty, Home, Sports, Books, Automotive.

Covers all edge cases from Spec §43:
- Empty, one-word, duplicate, HTML, multiple aspects, repeated aspects,
  opposite sentiments, aspect aliases, unknown aspects, irrelevant sentences,
  mixed sentiment, negation, intensifiers, long reviews, Unicode,
  punctuation, URLs, quoted text.

No downloads required. Fully deterministic with seed 42.
"""

from __future__ import annotations

import random
from typing import List

from productlens.schemas import ReviewRecord
from productlens.utils import stable_id


# ---------------------------------------------------------------------------
# Review templates per category
# ---------------------------------------------------------------------------

_ELECTRONICS_REVIEWS = [
    # Multi-aspect, opposite sentiments
    "The sound quality is excellent but the microphone is awful.",
    "The display is beautiful, but battery life is terrible. The keyboard feels very comfortable.",
    "Great camera quality during the day, but poor in low light.",
    "The build quality is excellent, but the buttons feel cheap.",
    "Battery lasts all day and the screen is gorgeous. However, the speaker is tinny.",
    "The processor is blazing fast but the fan noise is unbearable.",
    "Excellent picture quality on this TV. The remote is poorly designed though.",
    "Sound is crystal clear. Bass is weak compared to competitors.",
    "The touchscreen is very responsive. Battery drains too fast.",
    "Bluetooth connectivity works flawlessly. The charging cable feels flimsy.",
    # Single positive
    "Amazing sound quality, couldn't be happier with this purchase.",
    "The display resolution is absolutely stunning.",
    "Perfect keyboard for gaming, very responsive keys.",
    "This laptop runs incredibly fast for the price.",
    "The noise cancellation on these headphones is top notch.",
    # Single negative
    "The battery barely lasts two hours. Very disappointing.",
    "Screen started flickering after just one week of use.",
    "The WiFi reception is extremely poor in my house.",
    "Terrible audio quality, sounds like a tin can.",
    "The camera is completely useless in any indoor setting.",
    # Neutral / mixed
    "Decent laptop for the price. Nothing special but gets the job done.",
    "Average performance. The storage space is adequate.",
    "It works as expected. Not great, not terrible.",
    # Aliases (screen/display/panel/LCD, battery/battery life)
    "The screen is bright and vivid. Love the IPS panel.",
    "The LCD panel is just average for this price range.",
    "Battery backup is decent. Runtime is about 6 hours.",
    "The display quality could be better for a premium device.",
    # Contradictory opinions
    "Some say the battery is great, I found it barely adequate.",
    "The camera is excellent during the day but poor at night.",
    # Negation
    "The sound is not bad at all. Quite impressed.",
    "I wouldn't say the battery life is good.",
    "The keyboard doesn't feel cheap. It's actually well built.",
    # Intensifiers
    "The performance is incredibly disappointing.",
    "Absolutely phenomenal display quality!",
    "The speaker quality is somewhat acceptable.",
    # Irrelevant sentences
    "Amazon delivered it yesterday. The sound quality is good.",
    "My brother bought this for me. Package arrived Tuesday. The screen is decent.",
    "Got it on sale during Black Friday. The keyboard is comfortable.",
    # Long review with multiple aspects
    (
        "I've been using this laptop for three months now and have a lot to say. "
        "The display is absolutely gorgeous with vibrant colors and deep blacks. "
        "The keyboard has good travel and feels satisfying to type on. "
        "However, the trackpad is a bit small and sometimes misses gestures. "
        "Battery life is around 7 hours which is decent but not amazing. "
        "The speakers are louder than expected and sound reasonably clear. "
        "Build quality feels premium with the aluminum chassis. "
        "The webcam is terrible though, very grainy even in good lighting. "
        "Overall, it's a solid machine for the price point."
    ),
    (
        "After extensive testing of these wireless earbuds, here's my detailed review. "
        "Sound quality is exceptional with rich bass and clear mids. "
        "The noise cancellation is effective but not class-leading. "
        "Comfort is excellent - I can wear them for hours without fatigue. "
        "The microphone quality for calls is surprisingly good. "
        "Battery life meets the advertised 8 hours easily. "
        "The charging case is compact and the wireless charging works well. "
        "My only complaint is the touch controls are too sensitive. "
        "The Bluetooth connection drops occasionally when walking outside."
    ),
    # Repeated aspect mentions
    "The battery is great. I love how the battery handles intensive tasks. Best battery ever.",
    "Screen quality is good for movies but the screen glare is annoying outdoors.",
    # Additional electronics reviews
    "The mouse is ergonomic and comfortable. The scroll wheel is smooth.",
    "Terrible microphone quality. Everyone says I sound muffled on calls.",
    "The power adapter gets dangerously hot. Worried about fire risk.",
    "Webcam quality is surprisingly good for built-in. The mic picks up too much background noise though.",
    "Fast charging works as advertised. Battery goes from 0 to 50% in 30 minutes.",
    "The software is buggy and crashes frequently. Hardware is fine.",
    "Port selection is generous. Love having multiple USB-C ports.",
    "Screen resolution is sharp but the colors are oversaturated out of the box.",
    "Wireless range is excellent. Works through multiple walls without dropping.",
    "The cooling system keeps the laptop cool during gaming. Fan noise is acceptable.",
    "Terrible customer support experience. The product itself works great though.",
    "Storage speed is incredibly fast. Boot times are under 10 seconds.",
    "The external monitor support is limited. Only one external display at a time.",
    "Keyboard backlight is evenly distributed and has adjustable brightness.",
    "Volume quality is decent at medium levels but distorts at maximum.",
    "The hinge feels sturdy and holds the screen at any angle.",
    "Fingerprint reader is fast and reliable. Unlocks in under a second.",
    "The speakers lack bass but the mids and highs are clear.",
    "GPU performance exceeds expectations for this price range.",
    "The trackpad gestures work smoothly. Multi-finger swipes are responsive.",
]

_BEAUTY_REVIEWS = [
    "The moisturizer absorbs quickly and leaves skin feeling soft. The scent is too strong though.",
    "This foundation provides great coverage. The shade range could be better.",
    "Love the lash effect! Doesn't clump at all.",
    "The sunscreen leaves a white cast. Protection seems decent.",
    "Terrible product. Caused a breakout on my skin within two days.",
    "The texture is smooth and the formula is long-lasting. Packaging is flimsy.",
    "Perfect lip color, stays on all day without drying out.",
    "This serum changed my skincare routine. Dark spots faded in two weeks.",
    "The fragrance lasts for hours. Very elegant bottle design.",
    "Not worth the price. Generic formula with fancy packaging.",
    "The brush quality is excellent for applying powder.",
    "Skin feels hydrated all day. The pump dispenser is great.",
    "The cream is thick and takes forever to absorb.",
    "My hair feels so much softer after using this conditioner!",
    "The color payoff is amazing. Blends beautifully.",
    # Aliases (skin/complexion, moisturizer/cream/lotion)
    "The lotion feels lightweight. The cream texture is great for winter.",
    "My complexion looks better. The skin feels smooth and rejuvenated.",
    # Additional beauty reviews
    "The retinol serum caused initial irritation but results are visible after a month.",
    "Setting spray keeps makeup in place all day. Even through sweat.",
    "The eyebrow pencil tip is too thick for precise application.",
    "This dry shampoo is a lifesaver. Absorbs oil without white residue.",
    "The nail polish chips within two days. Very disappointing durability.",
    "Cleanser removes makeup effectively without stripping moisture.",
    "The body lotion has a pleasant scent that isn't overwhelming.",
    "Exfoliating scrub is too harsh for sensitive skin.",
    "The concealer provides excellent coverage for dark circles.",
    "Toner feels refreshing and tightens pores visibly.",
    "The face mask leaves skin feeling smooth but the fit is awkward.",
    "Hair oil adds great shine without making hair greasy.",
    "The palette colors are vibrant but the fallout is excessive.",
    "This BB cream provides light coverage with good SPF protection.",
    "Lip liner stays on through eating and drinking. Great formula.",
    # Edge case: punctuation
    "Great product!!! Love it!!! Would buy again!!!",
    # Unicode
    "Très bon produit! Works amazingly on my skin. 5★ quality!",
]

_HOME_REVIEWS = [
    "The vacuum cleaner has amazing suction power. The dustbin is too small though.",
    "Very quiet dishwasher. Energy efficient and cleans well.",
    "The blender motor is powerful. The lid doesn't seal properly.",
    "Love the air fryer! Makes crispy food with minimal oil. Timer is unreliable.",
    "The mattress is incredibly comfortable. Took a few nights to break in.",
    "Great coffee maker but the water reservoir is hard to clean.",
    "The light fixture looks modern. Installation was straightforward.",
    "This fan is whisper quiet and has good airflow coverage.",
    "The rug quality is disappointing for the price. Colors faded quickly.",
    "Excellent kitchen knife set. Stays sharp and handles well.",
    "The curtains block light effectively. Material feels cheap.",
    "Sturdy bookshelf, easy to assemble. Holds a lot of weight.",
    "The pillow is too flat. Not suitable for side sleepers.",
    "Air purifier works great! Noticed less dust within days.",
    "The toaster burns bread unevenly. Very inconsistent heating.",
    # Additional home reviews
    "The food processor chops evenly and is easy to disassemble for cleaning.",
    "Smart thermostat saved us money on energy bills. App is intuitive.",
    "The shower head water pressure is excellent. Installation was simple.",
    "Throw blanket is incredibly soft but sheds lint like crazy.",
    "The spice rack organizer saves so much counter space.",
    "Smoke detector is overly sensitive. Goes off when cooking regularly.",
    "The wine opener is elegant and works smoothly every time.",
    "Table lamp provides warm lighting. The dimmer switch is convenient.",
    "The door mat is durable but hard to clean.",
    "Ice maker produces ice quickly. The cubes are perfectly sized.",
    "The robot vacuum maps rooms efficiently but struggles with dark carpets.",
    "Kitchen towels are absorbent and dry quickly.",
    "The soap dispenser pump mechanism broke after a month.",
    "Drawer organizers fit perfectly. Everything has its place now.",
    "The candle scent fills the whole room. Burns evenly without tunneling.",
    # Delivery/service mentions (should be classified as non-product)
    "Delivery was fast. The product itself is decent.",
    "The seller was very responsive when I had issues. Chair is comfortable.",
]

_SPORTS_REVIEWS = [
    "These running shoes have excellent cushioning. The arch support could be better.",
    "The yoga mat is thick and grippy. Doesn't slip on hardwood floors.",
    "Great gym gloves, perfect fit. The velcro wears out quickly.",
    "The water bottle keeps drinks cold for 24 hours! Lid leaks sometimes.",
    "Resistance bands are durable and come in various strengths.",
    "The hiking backpack is spacious and comfortable. Zippers feel cheap.",
    "Excellent tennis racket. Great control and power.",
    "These swim goggles fog up within minutes. Very frustrating.",
    "The fitness tracker is accurate for step counting but heart rate is off.",
    "Good quality soccer ball. Holds air well and has nice touch.",
    "The bike seat is uncomfortable for long rides.",
    "Jump rope works perfectly. Handles have good grip.",
    "These compression socks are amazing for recovery.",
    "The punching bag is solid but the stand wobbles.",
    "Lightweight camping tent. Easy to set up but leaks in heavy rain.",
    # Additional sports reviews
    "The basketball has great grip on both indoor and outdoor courts.",
    "Running belt stays secure during sprints. Holds phone and keys comfortably.",
    "The foam roller is too soft for deep tissue work.",
    "Cycling gloves have excellent padding. No numbness after long rides.",
    "The climbing chalk bag is convenient. Drawstring closure works well.",
    "Badminton racket is lightweight and responsive. String tension is good.",
    "The exercise ball inflated easily and holds air well.",
    "Ski goggles provide excellent visibility. Anti-fog coating works.",
    "The dumbbell set takes up minimal space. Weight adjustment is smooth.",
    "Wrist wraps provide solid support for heavy lifts.",
    "The soccer shin guards are lightweight but protective.",
    "Swimming cap is too tight and uncomfortable after 20 minutes.",
    "The golf towel clips securely to the bag. Very absorbent.",
    "Rock climbing shoes have great grip but break-in period is painful.",
    "The kettlebell handle is smooth and comfortable. No hand tearing.",
]

_BOOKS_REVIEWS = [
    "Beautifully written. The plot twists kept me engaged throughout.",
    "The author's writing style is captivating but the ending felt rushed.",
    "Great reference book. The illustrations are helpful and clear.",
    "Too many grammatical errors. The content is interesting though.",
    "A must-read for anyone interested in machine learning.",
    "The paperback quality is good. Print is clear and easy to read.",
    "Story is predictable. Characters lack depth.",
    "Excellent research and well-documented arguments.",
    "The binding started falling apart after one reading.",
    "Informative and well-organized. Each chapter builds on the previous one.",
    "The translation could be better. Some nuance is lost.",
    "Perfect for beginners. The examples are practical and relevant.",
    "Overpriced for the amount of content provided.",
    "The appendix and index are thorough and useful.",
    "Couldn't put it down. Best thriller I've read this year.",
    # Additional reviews
    "The font size is perfect for reading without eye strain.",
    "Chapters are well-structured but examples could be more modern.",
    "The bibliography is comprehensive and points to great resources.",
    "The cover design is beautiful but the paper quality is mediocre.",
    "Engaging writing style that keeps you turning pages.",
    "Some chapters feel repetitive and could be condensed.",
    "The diagrams are helpful but too small to read clearly.",
    "The index is missing several key topics.",
    "Best textbook on this subject. The exercises are challenging.",
    "The ebook version has terrible formatting. Physical copy is fine.",
    "Footnotes are insightful and add valuable context.",
    "The prologue is slow but the story picks up dramatically.",
    "Page quality is thin and text bleeds through.",
    "The glossary is well-organized and easy to navigate.",
    "Several factual errors that should have been caught in editing.",
    "Companion website has excellent supplementary materials.",
    "The narrative voice is unique and refreshing.",
    "Too academic for casual readers. Dense prose throughout.",
    "The plot twist at the end was genuinely surprising.",
    "Would make an excellent gift for any science enthusiast.",
]

_AUTOMOTIVE_REVIEWS = [
    "The floor mats fit perfectly and are easy to clean. Material is durable.",
    "Phone mount holds securely even on bumpy roads. Suction cup is strong.",
    "Car vacuum cleaner has decent suction. The hose is too short.",
    "The dash cam video quality is excellent, especially at night.",
    "Seat covers look great but installation is a nightmare.",
    "LED headlight bulbs are much brighter than stock. Easy swap.",
    "The tire pressure gauge is accurate and easy to read.",
    "This car charger charges devices very slowly.",
    "The trunk organizer keeps everything tidy. Collapsible design is convenient.",
    "Windshield wipers streak after just a few uses.",
    "The steering wheel cover has a comfortable grip. Fits snugly.",
    "The air freshener scent is pleasant and long-lasting.",
    "These spark plugs improved my fuel efficiency noticeably.",
    "The GPS mount is unstable. Keeps falling off the windshield.",
    "Great oil filter. Easy to install and good quality.",
    # Additional reviews
    "The car cover fits well and protects against sun damage.",
    "Brake pads are noisy during the first few stops but quiet down after.",
    "The jump starter is compact but powerful enough for my SUV.",
    "Mirror adhesive didn't hold. Fell off within a week.",
    "These wiper fluid tablets work surprisingly well. Great value.",
    "The backup camera image quality is sharp and clear at night.",
    "Cup holder organizer is a great addition. Keeps drinks secure.",
    "The roof rack is sturdy but adds noticeable wind noise.",
    "OBD2 scanner works perfectly. Easy to diagnose engine codes.",
    "The sunshade is flimsy and doesn't stay in place.",
    "Mud flaps were easy to install. They protect well against road spray.",
    "Touch-up paint matches perfectly. You can barely see the scratch now.",
    "The key fob cover feels premium. Buttons still work perfectly through it.",
    "Cargo net keeps groceries from sliding around. Simple but effective.",
    "The aux cable audio quality is noticeably better than Bluetooth.",
]

# ---------------------------------------------------------------------------
# Edge case reviews (not tied to a specific category)
# ---------------------------------------------------------------------------

_EDGE_CASE_REVIEWS = [
    # Empty / very short (will be filtered during cleaning)
    ("Electronics", "", "empty_review"),
    ("Electronics", "Good", "one_word"),
    ("Electronics", "OK", "one_word_2"),
    ("Electronics", "  ", "whitespace_only"),
    # HTML content
    ("Electronics", "<b>Great</b> sound quality! <i>Love</i> the bass. <br/>Recommended.", "html_review"),
    ("Beauty", "The &amp; cream is &lt;amazing&gt; for dry skin.", "html_entities"),
    # URL in review
    ("Electronics", "Check out my full review at https://example.com/review/123 . The sound is great.", "url_review"),
    # Quoted text
    ("Electronics", 'My friend said "this is the best laptop ever" and I agree. The keyboard is excellent.', "quoted_review"),
    # Unicode
    ("Electronics", "Très bon son! Qualité excellente. The display is schön and überraschend.", "unicode_review"),
    ("Beauty", "この製品は素晴らしいです。The moisturizer is good.", "japanese_unicode"),
    # Duplicate content (same text, different product)
    ("Electronics", "The sound quality is excellent but the microphone is awful.", "duplicate_1"),
    # Heavy punctuation
    ("Electronics", "WOW!!!! AMAZING SOUND!!!! BEST EVER!!!!!", "heavy_punct"),
    ("Beauty", "Hmm... not sure about this one... texture is ok... I guess...", "ellipsis_review"),
    # Very long review
    ("Home", (
        "Let me tell you about my experience with this vacuum cleaner. "
        "First of all, the suction power is incredible. "
        "I've tried many vacuums over the years and this one stands out. "
        "The noise level is acceptable, not too loud but not silent either. "
        "The cord length is generous which means I don't have to keep unplugging. "
        "The dustbin capacity could be larger for bigger cleaning sessions. "
        "The filter system works well and is easy to clean. "
        "The attachments are varied and useful for different surfaces. "
        "The wheel design makes it easy to maneuver around furniture. "
        "The build quality feels solid and durable. "
        "Weight is reasonable, I can carry it upstairs without trouble. "
        "The handle is ergonomic and comfortable to grip. "
        "Customer service was helpful when I had a question about the warranty. "
        "Overall, this is the best vacuum I've owned. "
        "I would highly recommend it to anyone looking for a reliable cleaning solution."
    ), "long_review"),
    # Irrelevant content
    ("Electronics", "Amazon delivered it yesterday. Package arrived in good condition.", "irrelevant_only"),
    ("Electronics", "My brother bought this for his birthday.", "irrelevant_personal"),
    # Mixed sentiment with contradiction
    ("Electronics", "Half the reviewers love the battery, half hate it. I'm somewhere in between.", "contradiction_review"),
    # Multiple sentence punctuation types
    ("Electronics", "Is the battery good? Yes! The screen is amazing. So happy with this purchase!", "mixed_punct"),
    # Aspect at start and end
    ("Electronics", "Battery is the weakest point. Everything else is great except the battery.", "aspect_bookend"),
    # Numeric mentions
    ("Electronics", "Battery lasts 4.5 hours. Screen is 15.6 inches. Weighs only 3.2 lbs.", "numeric_review"),
    # Additional cross-category edge cases
    ("Electronics", "I can't believe how good the sound is. The noise cancellation is unreal.", "cant_contraction"),
    ("Electronics", "Returned it twice before getting one that works. Screen is beautiful when it works.", "returns_review"),
    ("Beauty", "My dermatologist recommended this. The ingredients list is clean.", "professional_rec"),
    ("Beauty", "Started using it three weeks ago and my skin cleared up completely!", "temporal_review"),
    ("Beauty", "The applicator brush is designed poorly. Product itself is fine.", "tool_vs_product"),
    ("Home", "My old blender lasted 10 years. This one broke in 3 months.", "comparison_old"),
    ("Home", "Works silently at night. Perfect for small apartments.", "use_case_specific"),
    ("Home", "The instruction manual is terrible. Took me 3 hours to assemble.", "documentation_issue"),
    ("Sports", "Used it during a marathon. Held up perfectly. No blisters.", "endurance_test"),
    ("Sports", "My trainer recommended these. Great for beginners and advanced users.", "expert_rec"),
    ("Books", "Read it in one sitting. Couldn't stop turning pages.", "engagement_review"),
    ("Books", "The Kindle version has formatting issues. Content is excellent.", "format_specific"),
    ("Automotive", "Installed it myself in 20 minutes. No tools needed.", "diy_install"),
    ("Automotive", "The OEM part costs three times more. This aftermarket option is just as good.", "value_comparison"),
    # More multi-aspect reviews
    ("Electronics", "Weight is light, screen is bright, but keyboard is cramped and speakers are tinny.", "quad_aspect"),
    ("Electronics", "Love the design, hate the price, neutral on the performance.", "three_sentiments"),
    ("Beauty", "Packaging is gorgeous but the product leaked during shipping. The cream itself is thick and rich.", "packaging_vs_product"),
    ("Home", "The timer works perfectly. The heating element is uneven. The cord is too short.", "triple_aspect_home"),
    ("Sports", "Grip is excellent, weight is balanced, but the handle coating peels after a month.", "multi_sports"),
    # Negation edge cases
    ("Electronics", "Not the worst laptop I've used. Not the best either.", "double_negation"),
    ("Electronics", "I don't think the battery is bad. It's actually quite decent.", "negated_negative"),
    ("Beauty", "Doesn't irritate my sensitive skin at all. Very gentle formula.", "negated_concern"),
    ("Home", "No complaints about the suction power. None whatsoever.", "no_complaints"),
    ("Sports", "Never had an issue with the stitching. Very well made.", "never_issue"),
    # Intensifier edge cases
    ("Electronics", "Extremely fast processor. Incredibly slow boot time. Fairly decent display.", "mixed_intensifiers"),
    ("Beauty", "Slightly disappointing texture. Remarkably effective ingredients.", "mixed_intensity_beauty"),
    # Service/delivery/seller mentions
    ("Electronics", "The seller shipped the wrong color. Product quality is good though.", "seller_issue"),
    ("Beauty", "Customer service was unhelpful but the product works great.", "service_mixed"),
    ("Home", "Packaging was damaged but the product inside was fine. Vacuum works perfectly.", "shipping_damage"),
    ("Sports", "Fast delivery! The product met my expectations.", "delivery_positive"),
    # Aspect type edge cases
    ("Electronics", "Amazon's return policy saved me. The laptop itself is mediocre.", "amazon_policy"),
    ("Home", "FedEx left it in the rain. The blender works fine surprisingly.", "carrier_issue"),
    # Very specific technical reviews
    ("Electronics", "The response time is 1ms. Input lag is nonexistent. Colors are accurate to sRGB.", "technical_review"),
    ("Electronics", "Supports USB-C PD 65W charging. HDMI 2.1 output works at 4K 120Hz.", "spec_heavy"),
    # Emotional/subjective reviews
    ("Electronics", "This laptop makes me happy every time I open it. Pure joy.", "emotional_positive"),
    ("Beauty", "I feel so much more confident with this foundation. Life changing!", "emotional_beauty"),
    # Comparison reviews
    ("Electronics", "Better than Sony but worse than Bose. Sound is clear but lacks depth.", "brand_comparison"),
    ("Home", "Dyson quality at half the price. The suction matches my V15.", "competitor_compare"),
    # Sarcastic/ironic
    ("Electronics", "Oh great, another laptop that overheats. What a surprise.", "sarcastic_review"),
    # Very positive
    ("Electronics", "Absolutely perfect in every way. Best purchase I've ever made. Five stars!", "superlative_positive"),
    # Very negative
    ("Electronics", "Complete waste of money. Worst product ever. Do not buy this garbage.", "superlative_negative"),
    # Balanced/detailed
    ("Beauty", "Pros: Long lasting, good coverage, nice shade range. Cons: Drying, expensive, small quantity.", "pros_cons_format"),
    # Temporal mentions
    ("Electronics", "Great for the first month. Started having issues after that. Now it barely works.", "temporal_decay"),
    ("Home", "Summer performance is excellent. Winter mode is adequate. Year-round reliability.", "seasonal_review"),
    # Gift reviews
    ("Electronics", "Bought this as a gift. My daughter loves it. The camera quality impressed her.", "gift_review"),
    ("Books", "Perfect gift for any aspiring programmer. Clear explanations.", "gift_book"),
    # Multiple products mentioned
    ("Electronics", "Pairs well with the matching keyboard. The mouse ergonomics are excellent.", "multi_product"),
    ("Automotive", "Works with both my sedan and my truck. Universal fit as described.", "multi_vehicle"),
    # Price-focused
    ("Electronics", "Overpriced for what you get. Similar features available for half the cost.", "price_negative"),
    ("Beauty", "Budget-friendly alternative to high-end brands. Quality is comparable.", "budget_review"),
    ("Home", "You get what you pay for. Premium quality at a premium price.", "price_quality"),
    # Durability-focused
    ("Electronics", "Still working perfectly after two years. Built to last.", "durability_positive"),
    ("Sports", "Fell apart during the first workout. Terrible build quality.", "durability_negative"),
    # Ambiguous sentiment
    ("Electronics", "It's fine. Does what it's supposed to. Nothing more, nothing less.", "ambiguous_neutral"),
    ("Beauty", "Different from what I expected. Not bad, just different.", "ambiguous_different"),
]


# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------

_PRODUCTS = {
    "Electronics": [
        ("ELEC001", "ELEC_P001", "Premium Wireless Headphones XR500"),
        ("ELEC002", "ELEC_P001", "Premium Wireless Headphones XR500 (Black)"),
        ("ELEC003", "ELEC_P002", "UltraBook Pro 15 Laptop"),
        ("ELEC004", "ELEC_P003", "Smart TV 55-inch 4K"),
        ("ELEC005", "ELEC_P004", "Wireless Earbuds Pro"),
        ("ELEC006", "ELEC_P005", "Mechanical Gaming Keyboard"),
    ],
    "Beauty": [
        ("BEAU001", "BEAU_P001", "HydraGlow Moisturizing Cream"),
        ("BEAU002", "BEAU_P002", "PerfectBase Foundation SPF30"),
        ("BEAU003", "BEAU_P003", "LashLux Volumizing Mascara"),
        ("BEAU004", "BEAU_P004", "ClearSkin Vitamin C Serum"),
    ],
    "Home": [
        ("HOME001", "HOME_P001", "PowerClean Vacuum Pro"),
        ("HOME002", "HOME_P002", "QuietWash Dishwasher"),
        ("HOME003", "HOME_P003", "SmartBlend Kitchen Blender"),
        ("HOME004", "HOME_P004", "CloudSleep Memory Mattress"),
    ],
    "Sports": [
        ("SPRT001", "SPRT_P001", "ProRun Ultra Running Shoes"),
        ("SPRT002", "SPRT_P002", "ZenStretch Yoga Mat"),
        ("SPRT003", "SPRT_P003", "FitTrack Smart Band"),
        ("SPRT004", "SPRT_P004", "Alpine Trek Hiking Pack"),
    ],
    "Books": [
        ("BOOK001", "BOOK_P001", "The Neural Network Chronicles"),
        ("BOOK002", "BOOK_P002", "Modern Machine Learning Handbook"),
        ("BOOK003", "BOOK_P003", "Whispers in the Algorithm"),
    ],
    "Automotive": [
        ("AUTO001", "AUTO_P001", "ProFit Custom Floor Mats"),
        ("AUTO002", "AUTO_P002", "NightVision Dash Cam 4K"),
        ("AUTO003", "AUTO_P003", "GripMount Phone Holder"),
    ],
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_synthetic_reviews(seed: int = 42) -> List[ReviewRecord]:
    """
    Generate deterministic synthetic reviews for smoke testing.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list of ReviewRecord
        Approximately 300 synthetic reviews.
    """
    rng = random.Random(seed)

    category_reviews = {
        "Electronics": _ELECTRONICS_REVIEWS,
        "Beauty": _BEAUTY_REVIEWS,
        "Home": _HOME_REVIEWS,
        "Sports": _SPORTS_REVIEWS,
        "Books": _BOOKS_REVIEWS,
        "Automotive": _AUTOMOTIVE_REVIEWS,
    }

    records: List[ReviewRecord] = []
    source = "synthetic"

    # Generate reviews from templates
    for category, templates in category_reviews.items():
        products = _PRODUCTS[category]

        for idx, text in enumerate(templates):
            # Assign to a product cyclically
            product = products[idx % len(products)]
            asin, parent_asin, product_title = product

            # Generate deterministic rating based on content sentiment
            rating = _estimate_rating(text, rng)
            timestamp = f"2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}T12:00:00Z"

            record = ReviewRecord(
                review_id=stable_id(source, asin, text, timestamp),
                product_id=asin,
                parent_product_id=parent_asin,
                category=category,
                title=_generate_title(text, rng),
                text=text,
                raw_text=text,
                clean_text="",  # Will be set during cleaning
                rating=rating,
                verified_purchase=rng.random() > 0.2,
                helpful_vote=rng.randint(0, 50),
                timestamp=timestamp,
                source=source,
            )
            records.append(record)

    # Add edge case reviews
    for category, text, label in _EDGE_CASE_REVIEWS:
        products = _PRODUCTS.get(category, _PRODUCTS["Electronics"])
        product = products[0]
        asin, parent_asin, product_title = product
        timestamp = f"2024-06-15T12:00:00Z"

        record = ReviewRecord(
            review_id=stable_id(source, asin, text, f"edge_{label}"),
            product_id=asin,
            parent_product_id=parent_asin,
            category=category,
            title=f"Edge case: {label}",
            text=text,
            raw_text=text,
            clean_text="",
            rating=_estimate_rating(text, rng),
            verified_purchase=True,
            helpful_vote=0,
            timestamp=timestamp,
            source=source,
        )
        records.append(record)

    # Duplicate some reviews to test dedup (deterministic selection)
    dup_indices = [0, 5, 10, 20]
    for di in dup_indices:
        if di < len(records):
            original = records[di]
            dup_record = ReviewRecord(
                review_id=stable_id(source, original.product_id, original.text, "dup"),
                product_id=original.product_id,
                parent_product_id=original.parent_product_id,
                category=original.category,
                title=original.title,
                text=original.text,
                raw_text=original.raw_text,
                clean_text="",
                rating=original.rating,
                verified_purchase=original.verified_purchase,
                helpful_vote=original.helpful_vote,
                timestamp=original.timestamp,
                source=source,
            )
            records.append(dup_record)

    rng.shuffle(records)
    return records


def _estimate_rating(text: str, rng: random.Random) -> float:
    """Estimate a star rating from review text content."""
    if not text or not text.strip():
        return float(rng.randint(1, 5))

    text_lower = text.lower()
    positive_words = [
        "excellent", "amazing", "great", "love", "perfect", "best",
        "stunning", "gorgeous", "outstanding", "fantastic", "incredible",
        "phenomenal", "brilliant", "superb",
    ]
    negative_words = [
        "terrible", "awful", "worst", "hate", "poor", "disappointing",
        "frustrating", "useless", "horrible", "cheap", "broken", "flimsy",
    ]

    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return float(rng.choice([4, 5]))
    elif neg_count > pos_count:
        return float(rng.choice([1, 2]))
    else:
        return float(rng.choice([3, 4]))


def _generate_title(text: str, rng: random.Random) -> str:
    """Generate a short review title from the review text."""
    positive_titles = [
        "Great product!", "Love it!", "Highly recommended",
        "Excellent quality", "Very satisfied", "Best purchase",
    ]
    negative_titles = [
        "Disappointed", "Not worth it", "Poor quality",
        "Could be better", "Save your money", "Returned it",
    ]
    neutral_titles = [
        "Decent product", "It's okay", "Average",
        "Fair for the price", "Mixed feelings", "As expected",
    ]

    text_lower = text.lower() if text else ""
    neg_indicators = ["terrible", "awful", "poor", "disappointing", "worst"]
    pos_indicators = ["excellent", "amazing", "great", "love", "perfect"]

    if any(w in text_lower for w in neg_indicators):
        return rng.choice(negative_titles)
    elif any(w in text_lower for w in pos_indicators):
        return rng.choice(positive_titles)
    else:
        return rng.choice(neutral_titles)
