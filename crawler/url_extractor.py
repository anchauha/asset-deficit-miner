#!/usr/bin/env python3
"""
Universal URL Content Extractor
================================

A flexible script to extract titles and content from URLs in any CSV file.
Outputs data in JSON or CSV format with proper encoding and metadata.

Usage:
    python url_extractor.py --csv_file <file> --url_column <column> [--output_file <file>] [--format <format>]
    
Arguments:
    --csv_file       Path to the CSV file containing URLs
    --url_column     Name of the column containing URLs
    --output_file    Optional: Output file path (defaults to input filename with new extension)
    --format         Optional: Output format 'json' or 'csv' (default: json)

Example:
    python url_extractor.py --csv_file "data/input/links.csv" --url_column "URL"
    python url_extractor.py --csv_file "data/input/urls.csv" --url_column "URL" --output_file "data/output/output.json"
    python url_extractor.py --csv_file "data/input/urls.csv" --url_column "URL" --output_file "data/output/output.csv" --format "csv"

Features:
- JSON and CSV output formats
- Proper UTF-8 encoding to handle special characters
- Structured JSON with metadata and extracted content
- Automatic backup creation
- Configurable content extraction
- Progress tracking
- Error handling and recovery
- Clean text extraction
"""

import csv
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from urllib.parse import urlparse
import re
import sys
import os
from pathlib import Path
import html
import json
import argparse

def clean_text(text):
    """
    Clean and normalize text content
    
    Args:
        text (str): Raw text to clean
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove excessive whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    
    # Remove control characters but keep basic punctuation
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_url_content(url, timeout=15):
    """
    Extract comprehensive content from a webpage URL
    
    Args:
        url (str): The URL to fetch content from
        timeout (int): Request timeout in seconds
    
    Returns:
        dict: Dictionary containing extracted content and metadata
    """
    try:
        # Headers to appear more like a regular browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        print(f"Extracting content from: {url}")
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Handle encoding properly
        response.encoding = response.apparent_encoding or 'utf-8'
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
        
        # Extract title
        title_tag = soup.find('title')
        title = clean_text(title_tag.get_text()) if title_tag else "No title found"
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if not meta_desc:
            meta_desc = soup.find('meta', attrs={'property': 'og:description'})
        description = clean_text(meta_desc.get('content', '')) if meta_desc else ""
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        keywords_meta = clean_text(meta_keywords.get('content', '')) if meta_keywords else ""
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            element.decompose()
        
        # Extract main content using multiple strategies
        content = ""
        
        # Strategy 1: Semantic HTML5 elements
        main_content = (soup.find('main') or 
                       soup.find('article') or 
                       soup.find('section', class_=re.compile(r'content|main|article|story|post', re.I)))
        
        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Strategy 2: Common content class patterns
            content_selectors = [
                '[class*="content"]',
                '[class*="article"]',
                '[class*="story"]',
                '[class*="post"]',
                '[class*="body"]',
                '[id*="content"]',
                '[id*="article"]',
                '.entry-content',
                '.post-content',
                '.article-content'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    content = content_div.get_text(separator=' ', strip=True)
                    break
            
            # Strategy 3: Extract all paragraph content
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text(separator=' ', strip=True) for p in paragraphs])
        
        # Clean the extracted content
        content = clean_text(content)
        
        # Generate keywords from URL structure
        url_keywords = []
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.split('/')
        for part in path_parts:
            if part and '-' in part:
                words = [word for word in part.split('-') if len(word) > 2]
                url_keywords.extend(words)
        
        # Combine keywords
        all_keywords = []
        if keywords_meta:
            all_keywords.extend([k.strip() for k in keywords_meta.split(',')])
        all_keywords.extend(url_keywords[:5])  # Limit URL keywords
        
        # Create summary (first 3 sentences or 300 characters)
        sentences = re.split(r'[.!?]+', content)
        summary_sentences = []
        summary_length = 0
        
        for sentence in sentences[:5]:  # Max 5 sentences
            sentence = sentence.strip()
            if sentence and summary_length + len(sentence) < 400:
                summary_sentences.append(sentence)
                summary_length += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_sentences)
        if summary and not summary.endswith('.'):
            summary += '.'
        
        # Limit content length for CSV storage
        content_full_length = len(content)
        content_truncated = content[:8000] if len(content) > 8000 else content
        
        return {
            'title': title,
            'description': description,
            'content': content_truncated,
            'summary': summary[:500],  # Limit summary
            'keywords': ', '.join(all_keywords[:8]),  # Limit keywords
            'content_length': content_full_length,
            'domain': parsed_url.netloc,
            'status': 'success'
        }
        
    except requests.exceptions.Timeout:
        error_msg = "Request timeout"
        print(f"Timeout error for {url}")
        return create_error_result(error_msg)
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Request error: {str(e)}"
        print(f"Request error for {url}: {e}")
        return create_error_result(error_msg)
        
    except Exception as e:
        error_msg = f"Parse error: {str(e)}"
        print(f"Unexpected error for {url}: {e}")
        return create_error_result(error_msg)

def create_error_result(error_msg):
    """Create a standardized error result"""
    return {
        'title': error_msg,
        'description': '',
        'content': '',
        'summary': '',
        'keywords': '',
        'content_length': 0,
        'domain': '',
        'status': 'error'
    }

def process_csv_file(csv_file_path, url_column, output_file_path=None, output_format='json'):
    """
    Process CSV file and extract content from URLs
    
    Args:
        csv_file_path (str): Path to input CSV file
        url_column (str): Name of column containing URLs
        output_file_path (str): Path to output file (optional)
        output_format (str): Output format ('json' or 'csv')
    
    Returns:
        bool: Success status
    """
    if output_file_path is None:
        # Create simple output filename based on input and format
        path = Path(csv_file_path)
        extension = '.json' if output_format == 'json' else '.csv'
        output_file_path = Path("data/output") / f"{path.stem}_extracted{extension}"
    
    # Ensure output directory exists
    Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file with proper encoding
    try:
        # Try different encodings
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding)
                print(f"Successfully read CSV with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not read CSV file with any supported encoding")
            
        print(f"Read {len(df)} rows from {csv_file_path}")
        print(f"Columns: {list(df.columns)}")
        
        if url_column not in df.columns:
            print(f"Error: Column '{url_column}' not found in CSV file")
            print(f"Available columns: {list(df.columns)}")
            return False
            
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False
    
    # Add new columns for extracted content
    new_columns = {
        'extracted_title': '',
        'extracted_description': '',
        'extracted_content': '',
        'extracted_summary': '',
        'extracted_keywords': '',
        'content_length': 0,
        'domain': '',
        'extraction_status': ''
    }
    
    for col, default_value in new_columns.items():
        if col not in df.columns:
            df[col] = default_value
            print(f"Added '{col}' column")
    
    # Process each URL
    successful_extractions = 0
    failed_extractions = 0
    
    for index, row in df.iterrows():
        url = row[url_column]
        
        # Skip if already processed (check if extracted_title exists)
        if (pd.notna(row.get('extracted_title', '')) and 
            str(row.get('extracted_title', '')).strip() and
            not str(row.get('extracted_title', '')).startswith('Request error') and
            not str(row.get('extracted_title', '')).startswith('Parse error')):
            print(f"Row {index + 1}: Already processed")
            successful_extractions += 1
            continue
        
        if pd.isna(url) or not str(url).strip():
            print(f"Row {index + 1}: Empty URL, skipping")
            continue
            
        print(f"\nProcessing row {index + 1}/{len(df)}: {url}")
        
        # Extract content
        result = extract_url_content(str(url))
        
        # Update DataFrame
        df.at[index, 'extracted_title'] = result['title']
        df.at[index, 'extracted_description'] = result['description']
        df.at[index, 'extracted_content'] = result['content']
        df.at[index, 'extracted_summary'] = result['summary']
        df.at[index, 'extracted_keywords'] = result['keywords']
        df.at[index, 'content_length'] = result['content_length']
        df.at[index, 'domain'] = result['domain']
        df.at[index, 'extraction_status'] = result['status']
        
        if result['status'] == 'success':
            successful_extractions += 1
            print(f"✓ Success: {result['title'][:60]}...")
        else:
            failed_extractions += 1
            print(f"✗ Failed: {result['title']}")
        
        # Be respectful to servers
        time.sleep(2)
    
    # Save the results with proper encoding
    try:
        if output_format == 'json':
            # Convert DataFrame to list of dictionaries for JSON output
            records = df.to_dict('records')
            
            # Clean up any NaN values for JSON serialization
            for record in records:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = ""
                    elif isinstance(value, (int, float)) and pd.isna(value):
                        record[key] = 0 if 'length' in key.lower() else ""
            
            # Create structured JSON output
            json_output = {
                "metadata": {
                    "extraction_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_file": csv_file_path,
                    "url_column": url_column,
                    "total_urls": len(df),
                    "successful_extractions": successful_extractions,
                    "failed_extractions": failed_extractions,
                    "success_rate": f"{(successful_extractions/(successful_extractions + failed_extractions)*100):.1f}%" if (successful_extractions + failed_extractions) > 0 else "0%"
                },
                "data": records
            }
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Saved enriched JSON to: {output_file_path}")
        else:
            # Save as CSV (original functionality)
            df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ Saved enriched CSV to: {output_file_path}")
        
        # Print summary
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total URLs processed: {successful_extractions + failed_extractions}")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Failed extractions: {failed_extractions}")
        print(f"Success rate: {(successful_extractions/(successful_extractions + failed_extractions)*100):.1f}%")
        
        if successful_extractions > 0:
            avg_length = df[df['content_length'] > 0]['content_length'].mean()
            total_content = df['content_length'].sum()
            print(f"Average content length: {avg_length:.0f} characters")
            print(f"Total content extracted: {total_content:,} characters")
        
        return True
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Universal URL Content Extractor - Extract titles and content from URLs in CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python url_extractor.py --csv_file "data/input/links.csv" --url_column "URL"
  python url_extractor.py --csv_file "data/input/urls.csv" --url_column "URL" --output_file "data/output/output.json"
  python url_extractor.py --csv_file "data/input/urls.csv" --url_column "URL" --output_file "data/output/output.csv" --format "csv"
        """
    )
    
    parser.add_argument(
        '--csv_file',
        default='data/input/sample.csv',
        required=False,
        help='Path to the CSV file containing URLs'
    )
    
    parser.add_argument(
        '--url_column',
        default='url',
        required=False,
        help='Name of the column containing URLs'
    )
    
    parser.add_argument(
        '--output_file',
        required=False,
        help='Output file path (defaults to data/output/<input_filename>_extracted.<format>)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'csv'],
        default='json',
        help='Output format: json or csv (default: json)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.csv_file):
        print(f"Error: File '{args.csv_file}' not found")
        sys.exit(1)
    
    print("Universal URL Content Extractor")
    print("=" * 50)
    print(f"Input file: {args.csv_file}")
    print(f"URL column: {args.url_column}")
    
    # Determine output file name if not provided
    if args.output_file is None:
        path = Path(args.csv_file)
        extension = '.json' if args.format == 'json' else '.csv'
        args.output_file = str(Path("data/output") / f"{path.stem}_extracted{extension}")
    
    print(f"Output file: {args.output_file}")
    print(f"Output format: {args.format.upper()}")
    print()
    
    # Create backup with simple naming
    backup_path = Path(args.csv_file).with_suffix('.bak')
    try:
        if not backup_path.exists():
            df_backup = pd.read_csv(args.csv_file)
            df_backup.to_csv(backup_path, index=False, encoding='utf-8-sig')
            print(f"Created backup: {backup_path}")
    except Exception as e:
        print(f"Warning: Could not create backup: {e}")
    
    # Process the file
    success = process_csv_file(args.csv_file, args.url_column, args.output_file, args.format)
    
    if success:
        print("\n🎉 Process completed successfully!")
    else:
        print("\n❌ Process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()