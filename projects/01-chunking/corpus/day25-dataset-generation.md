# Data Generation Guide: Using Cursor AI to Create Banking App Reviews

## Overview

This guide will help you use **Cursor AI** to generate a realistic banking app review dataset that matches the structure of the Netflix reviews dataset.

**Goal**: Generate ~50,000 banking app reviews (minimum 10,000 if time-constrained)

**Target Structure**: Must match Netflix dataset exactly (8 columns)

## Netflix Dataset Structure (Your Template)

Your generated banking reviews **MUST** have these exact columns:

| Column                 | Type          | Description         | Example                                | Can be NULL?     |
| ---------------------- | ------------- | ------------------- | -------------------------------------- | ---------------- |
| `reviewId`             | String (UUID) | Unique identifier   | `9594fc40-1280-49cc-b020-816ab174770c` | No               |
| `userName`             | String        | Reviewer name       | `John Smith`                           | Rare (~0.001%)   |
| `content`              | String        | Review text         | `"Great app but needs dark mode"`      | Rare (~0.004%)   |
| `score`                | Integer       | Rating 1-5          | `4`                                    | No               |
| `thumbsUpCount`        | Integer       | Helpful votes       | `3`                                    | No               |
| `reviewCreatedVersion` | String        | App version         | `2.15.3 build 4 12345`                 | Sometimes (~17%) |
| `at`                   | Datetime      | Review timestamp    | `2026-01-27 10:27:46`                  | No               |
| `appVersion`           | String        | Current app version | `2.15.3 build 4 12345`                 | Sometimes (~17%) |

**Rating Distribution** (try to match):

- 1 star: ~39%
- 2 stars: ~9%
- 3 stars: ~10%
- 4 stars: ~11%
- 5 stars: ~31%

## Step-by-Step Guide

### **Step 1: Understand the Banking App Context**

**App Name**: SecureBank Mobile  
**Category**: Finance & Banking  
**Key Features**:

- Account balance checking
- Money transfers (internal & external)
- Bill payments
- Mobile check deposit
- Budgeting tools
- Transaction history
- Card management (freeze/unfreeze)
- ATM locator
- Customer support chat

**Common Positive Themes**:

- Easy to use, secure, fast transfers, good UI, helpful features
- "Love the mobile deposit feature"
- "Transfers are instant"
- "Very secure and reliable"

**Common Negative Themes**:

- Login issues, app crashes, slow loading, missing features, poor customer service
- "App keeps crashing"
- "Can't log in after update"
- "Need fingerprint login"
- "Transfer limits too low"

### **Step 2: Set Up Your Environment**

1. **Open Cursor IDE**
2. **Create a new Python file**: `generate_banking_reviews.py`
3. **Create a prompt file**: `banking_prompt.txt` (we'll fill this in Step 3)

### **Step 3: Create Your Cursor AI Prompt**

Create a file called `banking_prompt.txt` with this content:

```
Generate realistic banking app reviews for "SecureBank Mobile" app.

REQUIREMENTS:
1. Generate 10,000 reviews (we'll run this 5 times to get 50,000 total)
2. Must have EXACTLY these 8 columns in this order:
   - reviewId (UUID format)
   - userName (realistic names, occasionally "Anonymous")
   - content (review text, 10-200 words, banking-specific)
   - score (1-5 integer)
   - thumbsUpCount (0-50, mostly 0-5)
   - reviewCreatedVersion (app version like "2.15.3 build 4 12345", 17% NULL)
   - at (timestamp between 2023-01-01 and 2026-01-27, format: YYYY-MM-DD HH:MM:SS)
   - appVersion (same format as reviewCreatedVersion, 17% NULL)

3. Rating distribution:
   - 1 star: 39% (negative reviews)
   - 2 stars: 9%
   - 3 stars: 10%
   - 4 stars: 11%
   - 5 stars: 31% (positive reviews)

4. Review content must be realistic and banking-specific:
   - Mention features: transfers, deposits, bill pay, security, UI, login
   - Include common complaints: crashes, login issues, slow, bugs
   - Include praise: easy to use, secure, fast, convenient
   - Vary length (some short "good app", some detailed paragraphs)
   - Include typos occasionally (realistic user-generated content)
   - Some reviews should mention specific issues or features

5. Output as CSV format with header row

BANKING APP CONTEXT:
- App: SecureBank Mobile
- Features: Account management, transfers, bill pay, mobile check deposit, budgeting, ATM locator
- Common issues: Login problems, app crashes, slow loading, transfer delays
- Common praise: Easy to use, secure, fast transfers, good UI

Generate the CSV data now.
```

### **Step 4: Use Cursor AI to Generate Data**

#### **Method 1: Direct Generation (Recommended)**

1. **Open Cursor IDE**
2. **Select the prompt** in `banking_prompt.txt`
3. **Press `Ctrl+K` (or `Cmd+K` on Mac)** to open Cursor AI
4. **Paste your prompt** and ask:

   ```
   Generate 10,000 banking app reviews as CSV following the exact requirements above.
   Save the output to banking_reviews_batch1.csv
   ```

5. **Wait for generation** (may take 2-5 minutes)

6. **Repeat 4 more times** to get different batches:
   - `banking_reviews_batch2.csv`
   - `banking_reviews_batch3.csv`
   - `banking_reviews_batch4.csv`
   - `banking_reviews_batch5.csv`

#### **Method 2: Python Script with Cursor AI Assistance**

Ask Cursor to help you write a Python script. Here's the minimal structure:

```python
import pandas as pd
import uuid
from datetime import datetime, timedelta
import random

def generate_review_id():
    """Generate UUID for reviewId"""
    # TODO: Return UUID string

def generate_user_name():
    """Generate realistic user names"""
    # TODO: Create list of first names and last names
    # TODO: Randomly combine them
    # TODO: 1% should be "Anonymous"

def generate_review_content(score):
    """Generate realistic review content based on score"""
    # TODO: Create lists of positive, negative, and neutral review templates
    # TODO: Return appropriate review based on score
    # TODO: Make reviews banking-specific

def generate_banking_reviews(n_reviews=10000):
    """Generate n banking app reviews"""

    reviews = []

    # TODO: Create rating distribution matching Netflix (39/9/10/11/31)
    # TODO: Shuffle ratings

    # TODO: Set date range (2023-01-01 to 2026-01-27)

    for i in range(n_reviews):
        # TODO: Generate each field
        # TODO: Handle 17% NULL for version fields
        # TODO: Create review dictionary
        # TODO: Append to reviews list

    return pd.DataFrame(reviews)

# Generate reviews
# TODO: Call generate_banking_reviews()
# TODO: Save to CSV
# TODO: Print statistics
```

**Important**: This is a **starter template**. You need to:

1. **Fill in all TODO sections**
2. **Create diverse review templates** (50+ templates per category)
3. **Make reviews realistic** (vary length, add typos, specific complaints)
4. **Ensure correct data types and formats**

### **Step 5: Validate Your Generated Data**

After generating, **always validate** your data:

```python
import pandas as pd

# Load your generated data
df = pd.read_csv('banking_reviews_batch1.csv')

# Validation checks
print("=== DATA VALIDATION ===\n")

# TODO: Check total number of reviews (should be >= 10,000)
# TODO: Check columns match expected list exactly
# TODO: Check rating distribution (should be close to 39/9/10/11/31)
# TODO: Check for duplicate reviewIds (should be 0)
# TODO: Check missing values (only in version columns, ~17%)
# TODO: Check data types are correct
# TODO: Sample and inspect 1-star and 5-star reviews
# TODO: Verify reviews are banking-specific (not Netflix content)

print("\n=== VALIDATION COMPLETE ===")
```

**Expected Output**:

- ✅ 10,000+ reviews
- ✅ Exact 8 columns in correct order
- ✅ Rating distribution close to 39/9/10/11/31
- ✅ No duplicate reviewIds
- ✅ ~17% missing values in version columns only
- ✅ Realistic banking-specific review content

### **Step 6: Combine Batches (If Generated Multiple)**

If you generated multiple batches, combine them:

```python
import pandas as pd

# TODO: Load all batch files
# TODO: Concatenate into single dataframe
# TODO: Remove any duplicate reviewIds
# TODO: Save final combined dataset
# TODO: Print final statistics (total reviews, rating distribution)
```

## Tips for High-Quality Generated Data

### 1. **Make Reviews Realistic**

**Bad** (too generic):

```
"Good app"
"Bad app"
"It's okay"
```

**Good** (specific and realistic):

```
"Love the instant transfer feature! Sent money to my friend and it arrived in seconds. The UI is clean and easy to navigate. Only wish there was a dark mode option."

"App crashes every time I try to deposit a check. I've tried reinstalling multiple times but the issue persists. Very frustrating when you need to deposit urgently!"

"Decent banking app. Does what it needs to do. Transfer feature works well but the budgeting tool could use more customization options. Overall satisfied."
```

### 2. **Include Variety**

- **Length**: Mix short (5 words) and long (100+ words) reviews
- **Tone**: Angry, happy, neutral, sarcastic
- **Topics**: Different features (login, transfers, UI, security, support)
- **Typos**: Occasional spelling mistakes (realistic user content)
- **Emojis**: Some users use emojis (👍, 😊, 😡, ⭐)

### 3. **Match Rating to Content**

Make sure review content matches the star rating:

- **1-2 stars**: Complaints, frustration, bugs, missing features
- **3 stars**: Mixed feedback, "it's okay", some issues
- **4-5 stars**: Praise, recommendations, positive experiences

### 4. **Banking-Specific Content**

Reviews should mention:

- **Features**: transfers, deposits, bill pay, budgeting, ATM locator
- **Security**: encryption, biometric login, fraud alerts
- **Issues**: login problems, crashes, slow loading, transfer limits
- **Praise**: convenience, speed, ease of use, reliability

## Troubleshooting

### **Problem 1: Cursor AI generates wrong format**

**Solution**: Be very specific in your prompt. Show an example row:

```
Example row:
9594fc40-1280-49cc-b020-816ab174770c,John Smith,"Great app!",5,2,2.15.3 build 4 12345,2026-01-27 10:27:46,2.15.3 build 4 12345
```

### **Problem 2: Reviews are too similar/repetitive**

**Solution**:

1. Generate smaller batches (1,000 at a time) with different prompts
2. Manually create 50+ diverse review templates
3. Use the Python script method for more control

### **Problem 3: Rating distribution doesn't match**

**Solution**: Explicitly specify the exact counts in your prompt:

```
Generate exactly:
- 3,900 reviews with score=1
- 900 reviews with score=2
- 1,000 reviews with score=3
- 1,100 reviews with score=4
- 3,100 reviews with score=5
Total: 10,000 reviews
```

### **Problem 4: Generated data has wrong data types**

**Solution**: Fix after generation:

```python
# TODO: Convert score to integer
# TODO: Convert thumbsUpCount to integer
# TODO: Convert 'at' to datetime
```

### **Problem 5: Cursor AI times out or doesn't respond**

**Solution**:

1. Generate smaller batches (1,000 reviews at a time)
2. Use the Python script method instead
3. Try during off-peak hours

## Example: Good vs Bad Generated Data

### ❌ **Bad Example**

```csv
reviewId,userName,content,score,thumbsUpCount,reviewCreatedVersion,at,appVersion
1,User1,Good,5,0,1.0,2024-01-01,1.0
2,User2,Bad,1,0,1.0,2024-01-02,1.0
3,User3,Okay,3,0,1.0,2024-01-03,1.0
```

**Problems**:

- reviewId not UUID format
- Generic usernames
- Content too short and generic
- All same version
- No missing values
- Not banking-specific

### ✅ **Good Example**

```csv
reviewId,userName,content,score,thumbsUpCount,reviewCreatedVersion,at,appVersion
9594fc40-1280-49cc-b020-816ab174770c,Sarah Johnson,"Love this banking app! The mobile deposit feature is a game changer. I can deposit checks from anywhere without going to the branch. Transfers are instant and the UI is very intuitive. Only minor issue is that the app sometimes takes a few seconds to load on my older phone, but overall excellent experience. Highly recommend!",5,3,2.18.4 build 6 45231,2025-06-15 14:23:11,2.18.4 build 6 45231
b6c60d1b-e33e-46b5-8e3f-7e16c8ea8d0a,Michael Chen,"App crashes constantly after the latest update. I can't even log in anymore - it just freezes on the loading screen. I've tried uninstalling and reinstalling multiple times but nothing works. This is extremely frustrating because I need to pay bills urgently. Please fix this ASAP!",1,12,,,2024-11-22 09:47:33,
cf0c7f42-775f-4c09-a6b5-e204a42f916e,Anonymous,decent app works fine for basic stuff but missing some features like fingerprint login and better budgeting tools,3,0,2.15.1 build 2 38492,2024-03-08 22:15:44,2.15.1 build 2 38492
```

**Why it's good**:

- Proper UUID format
- Realistic names (and one Anonymous)
- Detailed, banking-specific content
- Varied review lengths
- Some missing version data (realistic)
- Different timestamps
- Mentions specific banking features
