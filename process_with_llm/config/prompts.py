# prompts.py
# Configuration file for LLM prompts used in asset vs deficit language analysis

PROMPTS = {
    # 1. basic prompt for identifying asset vs deficit language
    "standard": (
"""Your goal is to identify asset-based versus deficit-based language within community-related texts. Read user-Input and highlight sections that reflect either an asset orientation (focusing on strengths, resources, or potential) or a deficit orientation (focusing on needs, problems, or limitations). For each identified segment, output a JSON object with a clear classification and reasoning.

Always return the output in a strict JSON format with the following keys: Type (Asset or Deficit), Text Span (the part of the text identified), and Reason (explaining why it fits that classification). If multiple examples exist in the text, output a JSON array of such objects.

Be direct, structured, and evaluative in tone. Keep your aim to support research or tool-building related to narrative analysis in social, educational, and community work contexts.

Input: {input_sentence}"""
    ),
    "semantic_extraction": (
"""Your goal is to identify asset-based versus deficit-based language within community-related texts using the provided framework.

# ABCD Framework for Classification:

## DEFICIT-BASED INDICATORS:
**Detection Patterns:**
- Lists what's wrong, problems, needs, deficiencies, or gaps
- Uses labels like "needy," "deficient," "at-risk," "disadvantaged," "underserved," "vulnerable".
- Creates dependency and positions outsiders as primary problem-solvers
- Focuses on scarcity mindset (what's missing, lacking, insufficient)

## ASSET-BASED INDICATORS:
**Detection Patterns:**
- Starts with strengths, existing gifts, capacities, resources, or potential
- Emphasizes community-led approach and internal assets
- Promotes inside-out thinking and collaborative partnerships
- Focuses on abundance mindset and existing resources

Now apply this framework to the Input text by identifying segments that reflect either an asset orientation (empowering, inside-out, strength-focused, etc) or a deficit orientation (harmful, top-down, dependency-creating, etc). For each identified segment, output a JSON object with a clear classification and reason. Always return the output in a strict JSON format with the following keys: Type (Asset, Deficit, or Mixed), Text Span (the part of the text identified), Reason (explaining why it fits that classification based on above INDICATORS, if Mixed then explain for both), semantic_asset_phrases (Array of words/phrases representing asset-based concepts, empty array if Type is "Deficit), semantic_deficit_phrases (Array of word/phrases representing deficit-based concepts, empty array if Type is "Asset"). If multiple segments exist in the text, output a JSON array of such objects.

# EXAMPLES:

**Asset Example:**
```json
{
    "Type": "Asset",
    "Text_Span": "leveraging community strengths and local leadership capacity",
    "Reason": "Emphasizes existing assets and community-led approach",
    "semantic_asset_phrases": ["community strengths", "local leadership capacity", "leveraging assets"],
    "semantic_deficit_phrases": []
}
```

**Deficit Example:**
```json
{
    "Type": "Deficit", 
    "Text_Span": "addressing the critical needs and service gaps in underserved populations",
    "Reason": "Focuses on problems and labels communities as underserved",
    "semantic_asset_phrases": [],
    "semantic_deficit_phrases": ["critical needs", "service gaps", "underserved populations"]
}
```

Be direct, structured, and evaluative in tone. Keep your aim to support research or tool-building related to narrative analysis in social, educational, and community work contexts.

Now analyze the following text:

Input: {input_sentence}"""
    ),
    "semantic_extraction2": (
"""
Your goal is to identify **asset-based** versus **deficit-based** language within **community-related texts** using the provided framework.

# Framework for Classification:

## DEFICIT-BASED INDICATORS:
**Detection Patterns:**
- Lists what's wrong, problems, needs, deficiencies, or gaps.
- Uses labels like "needy," "deficient," "at-risk," "disadvantaged," "underserved," "vulnerable".
- Creates dependency and positions outsiders as primary problem-solvers.
- Focuses on a scarcity mindset (what's missing, lacking, insufficient).

## ASSET-BASED INDICATORS:
**Detection Patterns:**
- Starts with strengths, existing gifts, capacities, resources, or potential.
- Emphasizes community-led approaches and internal assets.
- Promotes inside-out thinking and collaborative partnerships.
- Focuses on an abundance mindset and existing resources.

# Instructions for Output:
Apply the framework to the input text by identifying all segments that reflect either an asset or a deficit patterns. Your output must STRICTLY be a **JSON array** containing one or more objects, where each object represents an identified segment.

Each JSON object must have the following keys:
- "type": (String) CLASSIFICATION OF THE SEGMENT. MUST BE "Asset", "Deficit", OR "Mixed".
- "text_span": (String) THE EXACT PART OF THE TEXT THAT WAS IDENTIFIED.
- "reason": (String) A BRIEF EXPLANATION OF WHY THE TEXT FITS THE CLASSIFICATION, REFERENCING THE FRAMEWORK. FOR "Mixed" TYPES, EXPLAIN BOTH THE ASSET AND DEFICIT COMPONENTS.
- "semantic_asset_phrases": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT ASSET-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS "Deficit".
- "semantic_deficit_phrases": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT DEFICIT-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS "Asset".

# EXAMPLE:
**EXAMPLE community-related text:** "This neighborhood is a food desert, creating significant nutritional deficiencies. Building on the residents' passion for healthy living, local gardening groups are expanding their plots. While data shows only 40 percent of youth participate in after-school programs, the Youth Council has taken the initiative to design new programs."

**Output:**
```json
[
  {
    "type": "Deficit",
    "text_span": "This neighborhood is a food desert, creating significant nutritional deficiencies",
    "reason": "Uses 'food desert' label and focuses on nutritional deficiencies/problems",
    "semantic_asset_phrases": [],
    "semantic_deficit_phrases": ["food desert", "nutritional deficiencies"]
  },
  {
    "type": "Asset", 
    "text_span": "Building on the residents' passion for healthy living, local gardening groups are expanding their plots",
    "reason": "Emphasizes existing community assets (passion, local groups) and inside-out approach",
    "semantic_asset_phrases": ["residents' passion", "local gardening groups", "expanding"],
    "semantic_deficit_phrases": []
  },
  {
    "type": "Mixed",
    "text_span": "While data shows only 40 percent of youth participate in after-school programs, the Youth Council has taken the initiative to design new programs",
    "reason": "Deficit component highlights low participation rates as a problem. Asset component emphasizes Youth Council leadership and initiative-taking.",
    "semantic_asset_phrases": ["Youth Council", "taken the initiative", "design new programs"],
    "semantic_deficit_phrases": ["only 40 percent", "low participation"]
  }
]
```

Now analyze the following text:

Input: {input_sentence}"""
    ),
    "semantic_extraction_prompt3": (
"""
You are an expert linguist and content analyst specializing in asset-based and deficit-based language frameworks in **community context**.\n\n
# Your task:\n
Given a raw article, identify key segments of **asset-based** or **deficit-based** language.\n\n
# Framework for Identification:\n\n
## DEFICIT-BASED INDICATORS:\n
**Detection Patterns:**\n
- Lists what's wrong, problems, needs, deficiencies, or gaps.\n
- Uses labels similar to "needy," "deficient," "at-risk," "disadvantaged," "underserved," "vulnerable".\n
- Creates dependency and positions outsiders as primary problem-solvers.\n
- Focuses on a scarcity mindset (what's missing, lacking, insufficient).\n\n
**Important:**\n
- Descriptions of structural conditions should only be labeled "Deficit" if the language frames the community as passive, needy, or defined by those conditions. Factual or statistical statements are **not** deficit-based unless accompanied by deficit framing.\n
- Language that names real challenges or systemic issues is **not inherently deficit-based**. Only label it "Deficit" if it also removes agency, reinforces helplessness, or presents the community as broken or incapable.\n
- Calls to action or community-driven problem-solving should **not** be labeled "Deficit" unless the language also implies that solutions must come from outside the community.\n\n
## ASSET-BASED INDICATORS:\n
**Detection Patterns:**\n
- May start with strengths, existing gifts, capacities, resources, or potential.\n
- Emphasizes community-led approaches and internal assets.\n
- Promotes inside-out thinking and collaborative partnerships.\n
- Focuses on an abundance mindset and existing resources.\n\n
**Avoid:**\n
- Avoid confusing institutional branding or promotional language with authentic asset-based framing. These do not count unless grounded in specific internal strengths or actions.\n
- Avoid labeling segments as asset-based if the language downplays or erases structural inequities under the guise of positivity or resilience. Authentic asset-based framing should still acknowledge context.\n
- Positive or aspirational tone alone is not enough. For language to be asset-based, it must reflect existing strengths, actions, or capacities.\n\n

# Instructions for Output:\n
- Apply the framework to the article and only identify segments that clearly fit asset or deficit categories; ignore ambiguous text. Aim for three or more segments per article.\n\n
- Output in a structured JSON array for easy parsing where each object represents an identified segment.\n\n
Each JSON object must have the following fields:\n
- \"type\": (String) CLASSIFICATION OF THE SEGMENT. MUST BE \"Asset\", \"Deficit\", OR \"Mixed\".\n
- \"text_span\": (String) THE EXACT PART OF THE TEXT THAT WAS IDENTIFIED.\n
- \"reason\": (String) A BRIEF EXPLANATION OF WHY THE TEXT FITS THE CLASSIFICATION, REFERENCING THE FRAMEWORK. FOR \"Mixed\" TYPES, EXPLAIN BOTH THE ASSET AND DEFICIT COMPONENTS.\n
- \"semantic_asset_phrases\": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT ASSET-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS \"Deficit\".\n
- \"semantic_deficit_phrases\": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT DEFICIT-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS \"Asset\".\n\n

# One-Shot Example:\n
**Article Snippet:** \"This neighborhood is a food desert, creating significant nutritional deficiencies. Building on the residents' passion for healthy living, local gardening groups are expanding their plots. While data shows only 40 percent of youth participate in after-school programs, the Youth Council has taken the initiative to design new programs.\"\n\n

**Output:**\n
```json
[
{{
    "type": "Deficit",
    "text_span": "neighborhood is a food desert, creating significant nutritional deficiencies",
    "reason": "This is deficit-based language. Instead of neutrally stating a lack of stores, it uses the loaded label 'food desert' and frames the community solely by its 'nutritional deficiencies,' focusing entirely on the problem.",
    "semantic_asset_phrases": [],
    "semantic_deficit_phrases": [
    "food desert",
    "significant nutritional deficiencies"
    ]
}},
{{
    "type": "Asset",
    "text_span": "Building on the residents' passion for healthy living, local gardening groups are expanding their plots",
    "reason": "This segment's language is asset-based because it starts with internal community strengths ('residents' passion') and highlights resident-led action ('local gardening groups are expanding'), positioning them as capable problem-solvers.",
    "semantic_asset_phrases": [
    "Building on the residents' passion",
    "healthy living",
    "expanding their plots"
    ],
    "semantic_deficit_phrases": []
}},
{{
    "type": "Mixed",
    "text_span": "data shows only 40 percent of youth participate in after-school programs, the Youth Council has taken the initiative to design new programs",
    "reason": "The language is mixed. It uses the word 'only' to frame the 40 percent statistic as a deficit. However, the primary focus immediately shifts to an asset: the agency and capacity of the 'Youth Council' taking the 'initiative' to solve the problem themselves.",
    "semantic_asset_phrases": [
    "has taken the initiative",
    "design new programs"
    ],
    "semantic_deficit_phrases": [
    "only 40 percent of youth participate"
    ]
}}
]
```

Now **identify and output key segments in JSON format from the following raw article while focusing strictly on the LANGUAGE, FRAMING, and RHETORIC used:**\n\n

**Raw Article:**\n'''\n{input_sentence}\n'''\n
"""
    ),
    "semantic_extraction_prompt4": (
"""\n
You are an expert linguist and content analyst specializing in asset-based and deficit-based language frameworks in **community context**.\n\n
# Your task:\n
Given a raw article, identify key segments of **asset-based** or **deficit-based** language.\n\n
# Framework for Identification:\n\n
## DEFICIT-BASED INDICATORS:\n
**Detection Patterns:**\n
- Lists what's wrong, problems, needs, deficiencies, or gaps.\n
- Uses labels similar to "needy," "deficient," "at-risk," "disadvantaged," "underserved," "vulnerable".\n
- Creates dependency and positions outsiders as primary problem-solvers.\n
- Focuses on a scarcity mindset (what's missing, lacking, insufficient).\n\n
**Important:**\n
- Descriptions of structural conditions should only be labeled "Deficit" if the language frames the community as passive, needy, or defined by those conditions. Factual or statistical statements are **not** deficit-based unless accompanied by deficit framing.\n
- Language that names real challenges or systemic issues is **not inherently deficit-based**. Only label it "Deficit" if it also removes agency, reinforces helplessness, or presents the community as broken or incapable.\n
- Calls to action or community-driven problem-solving should **not** be labeled "Deficit" unless the language also implies that solutions must come from outside the community.\n\n
## ASSET-BASED INDICATORS:\n
**Detection Patterns:**\n
- May start with strengths, existing gifts, capacities, resources, or potential.\n
- Emphasizes community-led approaches and internal assets.\n
- Promotes inside-out thinking and collaborative partnerships.\n
- Focuses on an abundance mindset and existing resources.\n\n
**Avoid:**\n
- Avoid confusing institutional branding or promotional language with authentic asset-based framing. These do not count unless grounded in specific internal strengths or actions.\n
- Avoid labeling segments as asset-based if the language downplays or erases structural inequities under the guise of positivity or resilience. Authentic asset-based framing should still acknowledge context.\n
- Positive or aspirational tone alone is not enough. For language to be asset-based, it must reflect existing strengths, actions, or capacities.\n\n
# Instructions for Output:\n
- Apply the framework to the article and only identify segments that clearly fit asset or deficit categories; ignore ambiguous text. Aim for three or more segments per article.\n\n
- Output in a structured JSON array for easy parsing where each object represents an identified segment.\n\n
Each JSON object must have the following fields:\n
- \"type\": (String) CLASSIFICATION OF THE SEGMENT. MUST BE \"Asset\", \"Deficit\", OR \"Mixed\".\n
- \"text_span\": (String) THE EXACT PART OF THE TEXT THAT WAS IDENTIFIED.\n
- \"reason\": (String) A BRIEF EXPLANATION OF WHY THE TEXT FITS THE CLASSIFICATION, REFERENCING THE FRAMEWORK. FOR \"Mixed\" TYPES, EXPLAIN BOTH THE ASSET AND DEFICIT COMPONENTS.\n
- \"semantic_asset_phrases\": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT ASSET-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS \"Deficit\".\n
- \"semantic_deficit_phrases\": (Array of strings) AN ARRAY OF WORDS/PHRASES FROM THE SPAN THAT REPRESENT DEFICIT-BASED CONCEPTS. SHOULD BE AN EMPTY ARRAY [] IF THE TYPE IS \"Asset\".\n\n
Now **identify and output key segments in JSON format from the following raw article while focusing strictly on the LANGUAGE, FRAMING, and RHETORIC used:**\n\n
# **Raw Article:**\n'''\n{input_sentence}\n'''\n
"""
    ),
    "asset_reframing1": (
"""
You are a community development expert specializing in asset-based community development (ABCD). Your task is to transform deficit-based language into asset-based language while maintaining the core meaning and context.

# Asset-Based Language Principles:

## FROM Deficit Language TO Asset Language:
- Problems → Opportunities for growth
- Needs → Existing capacities to build upon  
- Services for people → People as co-producers
- Clients/beneficiaries → Community members/partners
- Interventions → Collaborative initiatives
- At-risk populations → Communities with untapped potential
- Service delivery → Community mobilization
- Professional expertise → Shared knowledge and wisdom

## Key Transformation Strategies:
1. **Lead with strengths**: Start with what exists, not what's missing
2. **Community agency**: Position residents as primary actors
3. **Inside-out thinking**: Build from community assets outward
4. **Collaborative partnerships**: Replace service delivery with cooperation
5. **Asset mapping**: Identify existing resources and connections

# Your Task:
Transform the provided text by:
1. Identifying deficit-based language patterns
2. Rewriting them using asset-based alternatives
3. Maintaining factual accuracy and context
4. Ensuring the reframed version is authentic and practical

Provide your response in JSON format:
```json
{
  "original_text": "the complete original text",
  "reframed_text": "your asset-based reframing",
  "transformations": [
    {
      "original_phrase": "deficit phrase identified",
      "reframed_phrase": "asset-based alternative", 
      "reasoning": "explanation of the transformation"
    }
  ],
  "asset_elements_added": ["list", "of", "new", "asset-based", "elements"],
  "overall_shift": "description of the overall narrative change"
}
```

Text to reframe: {input_sentence}"""
    )
}

# Configuration for model-specific settings
MODEL_CONFIGS = {
    "huggingface": {
        "default_model": "meta-llama/Llama-2-7b-chat-hf",
        "default_quantization": "auto",
        "default_max_length": 4096,
        "default_temperature": 0.7,
        "default_top_p": 0.9,
        "cache_dir": "/N/scratch/ankichau/hf_cache"
    },
    "ollama": {
        "default_model": "llama4:17b-scout-16e-instruct-fp16",
        "default_timeout": 30
    }
}

# Environment configuration
ENV_CONFIG = {
    "hf_token": "NAAAAAAYYYY", # Replace with your actual Hugging Face token
    "cache_base": "/N/scratch/ankichau/hf_cache"
}