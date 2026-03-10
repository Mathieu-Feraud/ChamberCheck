"""
Media preprocessing module for ChamberCheck.

Extracts and analyzes media content from Reddit posts:
- Images: Uses vision LLM to describe and extract text
- Links: Scrapes external links and summarizes content
- Videos: Detects video URLs for reference

Core functions to process raw scraped data and add 'extracted_media' field.
"""

import json
import os
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

from ..analysis.openai_provider import OpenAIProvider
from ..config import Config
from ..constants import (
    MEDIA_USER_AGENT,
    MEDIA_BROWSER_USER_AGENT,
    MEDIA_MAX_RETRIES,
    MEDIA_INITIAL_RETRY_DELAY,
    MEDIA_MAX_LINK_CONTENT_LENGTH,
    MEDIA_MAX_SUMMARY_WORDS,
    MEDIA_FALLBACK_TEXT_LENGTH,
    MEDIA_REQUEST_TIMEOUT,
    MEDIA_VISION_ANALYSIS_PROMPT
)


def get_reddit_post_details(post_id: str) -> Optional[Dict]:
    """Fetch full post details from Reddit JSON API to get media URLs."""
    try:
        url = f"https://www.reddit.com/comments/{post_id}.json"
        headers = {'User-Agent': MEDIA_USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=MEDIA_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        if data and len(data) > 0:
            post_data = data[0]['data']['children'][0]['data']
            return post_data
        
        return None
    except Exception as e:
        print(f"  ⚠️  Error fetching post details: {e}")
        return None


def extract_image_url(post_data: Dict) -> Optional[str]:
    """Extract image URL from Reddit post data."""
    try:
        # Check for direct image URL
        if post_data.get('url') and 'reddit.com' not in post_data['url']:
            url = post_data['url']
            # Exclude Reddit-hosted video URLs (v.redd.it) — handled by extract_video_url
            if 'v.redd.it' in url:
                return None
            # Only return if it's an absolute URL
            if url.startswith('http'):
                return url
        
        # Check for preview object
        preview = post_data.get('preview')
        if preview and 'images' in preview:
            images = preview['images']
            if images and len(images) > 0:
                url = images[0]['source']['url'].replace('&amp;', '&')
                # Validate it's an absolute URL
                if url.startswith('http'):
                    return url
        
        # Check for gallery posts
        gallery_data = post_data.get('gallery_data')
        if gallery_data and 'items' in gallery_data:
            items = gallery_data['items']
            if items and len(items) > 0:
                media_id = items[0].get('media_id')
                if media_id:
                    media_metadata = post_data.get('media_metadata', {})
                    if media_id in media_metadata:
                        media_info = media_metadata[media_id]
                        if media_info.get('type') == 'image':
                            image_source = media_info.get('s', {}).get('x')
                            if image_source and image_source.startswith('http'):
                                return image_source
        
        return None
    except Exception as e:
        print(f"  ⚠️  Error extracting image URL: {e}")
        return None


def extract_video_url(post_data: Dict) -> Optional[str]:
    """Extract video URL from Reddit post data."""
    try:
        url = post_data.get('url', '')
        
        # YouTube
        if 'youtube.com' in url or 'youtu.be' in url:
            return url
        
        # Vimeo
        if 'vimeo.com' in url:
            return url
        
        # Twitch
        if 'twitch.tv' in url:
            return url
        
        # Streamable
        if 'streamable.com' in url:
            return url
        
        # Reddit video
        if post_data.get('media'):
            media = post_data['media']
            if 'reddit_video' in media:
                return media['reddit_video'].get('fallback_url')
        
        return None
    except Exception as e:
        print(f"  ⚠️  Error extracting video URL: {e}")
        return None


def fetch_and_extract_link_content(url: str) -> Optional[str]:
    """Fetch and extract text content from an external link."""
    try:
        headers = {
            'User-Agent': MEDIA_BROWSER_USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        response = requests.get(url, headers=headers, timeout=MEDIA_REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        # Filter out common JS-blocked or placeholder pages
        lower_text = text.lower()
        if "javascript is not available" in lower_text or "enable javascript" in lower_text:
            return None
        if "we’ve detected that javascript is disabled" in lower_text or "we've detected that javascript is disabled" in lower_text:
            return None
        
        return text[:MEDIA_MAX_LINK_CONTENT_LENGTH] if text else None
    except Exception as e:
        print(f"  ⚠️  Error fetching link content: {e}")
        return None


def summarize_content(text: str, llm_provider: OpenAIProvider, max_words: int = MEDIA_MAX_SUMMARY_WORDS) -> str:
    """Summarize long text content using LLM."""
    if not text or len(text.split()) <= max_words:
        return text
    
    try:
        response = llm_provider.analyze_with_text(
            prompt=f"Summarize the following text in {max_words} words or less:\n\n{text}"
        )
        return response
    except Exception as e:
        print(f"  ⚠️  Error summarizing content: {e}")
        return text[:MEDIA_FALLBACK_TEXT_LENGTH]  # Return truncated original




def analyze_image_with_llm(image_url: str, llm_provider: OpenAIProvider) -> Dict:
    """Analyze an image using LLM vision capabilities with retry logic."""
    prompt = MEDIA_VISION_ANALYSIS_PROMPT
    
    max_retries = MEDIA_MAX_RETRIES
    retry_delay = MEDIA_INITIAL_RETRY_DELAY
    
    for attempt in range(max_retries):
        try:
            # Call OpenAI vision model
            response = llm_provider.analyze_with_vision(
                prompt=prompt,
                image_url=image_url
            )
            break  # Success, exit retry loop
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                if attempt < max_retries - 1:
                    print(f"  ⏳ Rate limit hit, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                # Rate limit exhausted
                return {"error": err_str, "description": "Rate limit exceeded after retries"}
            # Non-retryable error (e.g. 400 bad URL) — record and move on
            print(f"  ⚠️  Vision API error (skipping): {e}")
            return {"error": err_str, "description": "Vision API error"}
    
    try:
        # Try to parse as JSON - handle markdown code blocks
        try:
            # Remove markdown code blocks if present
            cleaned = response.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]  # Remove ```json
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]  # Remove ```
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]  # Remove closing ```
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # If not valid JSON, wrap in description
            return {
                "description": response,
                "text_content": "",
                "media_type": "unknown",
                "platform": "none",
                "extracted_data": {}
            }
    
    except Exception as e:
        print(f"  ⚠️  Error analyzing image: {e}")
        return {
            "error": str(e),
            "description": "Failed to analyze image"
        }


def process_posts(input_file: str, post_ids: list = None, force_reprocess: bool = False) -> Dict:
    """Process raw data file to extract media information.
    
    Modifies the input file in place by adding extracted_media to posts.
    
    Args:
        input_file: Path to raw data JSON file
        post_ids: Optional list of post IDs to process (if None, process all)
        force_reprocess: If True, reprocess posts even if extracted_media exists
    
    Returns:
        dict: Summary statistics about processing
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print("ChamberCheck - Media Preprocessing")
    print("=" * 70)
    print(f"File: {input_file}")
    if post_ids:
        print(f"Filtering to post IDs: {post_ids}")
    
    # Load raw data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both dict with 'posts' key and list format
    if isinstance(data, dict):
        posts = data.get('posts', [])
        original_data = data
    else:
        posts = data
        original_data = posts
    
    print(f"\n📊 Found {len(posts)} posts")
    
    # Filter posts if post_ids specified (for processing only, keep original for saving)
    posts_to_process = posts
    if post_ids:
        posts_to_process = [p for p in posts if p.get('post_id') in post_ids]
        print(f"   Filtered to {len(posts_to_process)} matching posts")
    
    # Initialize LLM provider (optional - will skip analysis if not available)
    config = Config()
    api_key = config.get('llm.api_key') or os.getenv('OPENAI_API_KEY')
    llm_provider = None
    
    if api_key:
        llm_provider = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
        print("✓ OpenAI API key found - will analyze media")
    else:
        print("ℹ️  OpenAI API key not configured - will extract media URLs only")
    
    # Process posts - track post IDs by processing outcome
    processed_ids = []  # Successfully processed with LLM
    already_processed_ids = []  # Already processed (extracted_media exists)
    no_processing_needed_ids = []  # Self posts or no media found
    error_ids = []  # Errors during processing
    image_posts = 0
    
    for i, post in enumerate(posts_to_process, 1):
        post_id = post.get('post_id')
        title = post.get('title', '')[:50]
        
        print(f"\n[{i}/{len(posts_to_process)}] {title}...")
        
        # Delete existing extracted_media if force_reprocess is True
        if force_reprocess and 'extracted_media' in post:
            del post['extracted_media']
            print(f"  🔄 Deleted existing extracted_media (force reprocess)")
        
        # Skip if already processed
        if 'extracted_media' in post and not force_reprocess:
            print(f"  ✓ Already processed")
            already_processed_ids.append(post_id)
            continue
        
        # Skip if it's a self post (text post)
        if post.get('metadata', {}).get('is_self'):
            print(f"  ✓ Self post (text only, no media)")
            post['extracted_media'] = {
                'status': 'no_processing_needed',
                'reason': 'self_post',
                'processed_at': datetime.now().isoformat()
            }
            no_processing_needed_ids.append(post_id)
            continue
        
        # Check for external link FIRST (for link posts)
        # Use the external_url from our scraped metadata if available
        link_url = None
        image_url = None
        video_url = None
        external_url = post.get('metadata', {}).get('external_url')

        if external_url:
            external_url = external_url.strip()

            # Ignore relative reddit paths
            if external_url.startswith('/r/'):
                external_url = None

        if external_url:
            lower_url = external_url.lower()
            is_image = (
                lower_url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
                or 'i.redd.it' in lower_url
            )
            is_video = (
                lower_url.endswith(('.mp4', '.webm', '.mov'))
                or 'v.redd.it' in lower_url
            )

            if is_image:
                image_url = external_url
            elif is_video:
                video_url = external_url
            elif 'reddit.com' not in lower_url:
                link_url = external_url

        # If we found an external link, process it and skip fetching post details
        if link_url or image_url or video_url:
            # Will be processed below
            pass
        else:
            # Fetch full post details to get image URL for embedded media
            print(f"  🔍 Fetching post details...")
            post_details = get_reddit_post_details(post_id)
            
            if not post_details:
                print(f"  ⚠️  Could not fetch post details")
                post['extracted_media'] = {
                    'status': 'failed',
                    'reason': 'post_details_unavailable',
                    'processed_at': datetime.now().isoformat()
                }
                error_ids.append(post_id)
                continue
            
            # Extract image URL
            image_url = extract_image_url(post_details)
            
            # Try video if no image found
            video_url = None
            if not image_url:
                video_url = extract_video_url(post_details)
        
        # For v.redd.it links, try to resolve to a direct video URL if possible
        if video_url and 'v.redd.it' in video_url:
            post_details = get_reddit_post_details(post_id)
            if post_details:
                resolved_video = extract_video_url(post_details)
                if resolved_video:
                    video_url = resolved_video
        
        if not image_url and not video_url and not link_url:
            print(f"  ℹ️  No image, video, or external link found")
            post['extracted_media'] = {
                'status': 'no_processing_needed',
                'reason': 'no_media_found',
                'processed_at': datetime.now().isoformat()
            }
            no_processing_needed_ids.append(post_id)
            continue
        
        # Process image
        if image_url:
            print(f"  🖼️  Found image: {image_url[:60]}...")
            image_posts += 1
            
            # Analyze image with LLM if available
            analysis = None
            if llm_provider:
                print(f"  🤖 Analyzing with vision LLM...")
                analysis = analyze_image_with_llm(image_url, llm_provider)
                
                # Summarize if needed
                text_content = analysis.get('text_content') if isinstance(analysis, dict) else None
                if text_content and isinstance(text_content, str) and len(text_content.split()) > 300:
                    print(f"  📝 Summarizing long text...")
                    analysis['text_content'] = summarize_content(text_content, llm_provider)
                    analysis['is_summarized'] = True
            
            # Add extracted information to post
            post['extracted_media'] = {
                'media_type': 'image',
                'image_url': image_url,
                'analysis': analysis,
                'processed_at': datetime.now().isoformat()
            }
            
            print(f"  ✓ Extracted image info")
            text_content = analysis.get('text_content') if isinstance(analysis, dict) else None
            if text_content and isinstance(text_content, str):
                text_len = len(text_content)
                summary_note = " (summarized)" if isinstance(analysis, dict) and analysis.get('is_summarized') else ""
                print(f"    └─ {text_len} chars{summary_note}")
            
            # Track based on whether LLM was used
            if llm_provider and analysis:
                processed_ids.append(post_id)
            else:
                no_processing_needed_ids.append(post_id)
        
        # Process video
        elif video_url:
            print(f"  🎥 Found video: {video_url[:60]}...")
            
            post['extracted_media'] = {
                'media_type': 'video',
                'video_url': video_url,
                'processed_at': datetime.now().isoformat()
            }
            
            print(f"  ✓ Found video")
            no_processing_needed_ids.append(post_id)
        
        # Process external link
        elif link_url:
            print(f"  🔗 Found external link: {link_url[:60]}...")
            
            # Fetch and extract content from link
            print(f"  🔍 Fetching link content...")
            link_content = fetch_and_extract_link_content(link_url)
            
            if link_content:
                # Summarize if needed (if LLM available)
                summary = None
                topic_label = None
                if llm_provider:
                    print(f"  📝 Summarizing link content...")
                    summary = summarize_content(link_content, llm_provider)
                    try:
                        topic_prompt = (
                            "Extract a concise 2-6 word topic label that summarizes the main subject. "
                            "Return ONLY the label.\n\n"
                            f"Text:\n{summary}"
                        )
                        topic_label = llm_provider.analyze_with_text(topic_prompt)
                        if isinstance(topic_label, str):
                            topic_label = topic_label.strip().strip('"').strip("'")
                            topic_label = " ".join(topic_label.split())
                            words = topic_label.split()[:6]
                            topic_label = " ".join(words) if words else None
                    except Exception as e:
                        print(f"  ⚠️  Error extracting topic label: {e}")
                else:
                    # Without LLM, just truncate if too long
                    summary = link_content[:MEDIA_FALLBACK_TEXT_LENGTH] if len(link_content) > MEDIA_FALLBACK_TEXT_LENGTH else link_content
                
                post['extracted_media'] = {
                    'media_type': 'link',
                    'link_url': link_url,
                    'extracted_text': summary,
                    'topic': topic_label,
                    'is_summarized': llm_provider is not None and len(link_content.split()) > 300,
                    'processed_at': datetime.now().isoformat()
                }
                
                print(f"  ✓ Extracted link info")
                if summary:
                    print(f"    └─ {len(summary)} chars")
                
                # Track based on whether LLM was used
                if llm_provider:
                    processed_ids.append(post_id)
                else:
                    no_processing_needed_ids.append(post_id)
            else:
                print(f"  ⚠️  Could not extract content from link")
                post['extracted_media'] = {
                    'status': 'failed',
                    'reason': 'link_content_unavailable',
                    'link_url': link_url,
                    'processed_at': datetime.now().isoformat()
                }
                error_ids.append(post_id)
    
    # Save back to original file (always save all posts, not just filtered ones)
    json_to_save = original_data
    
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(json_to_save, f, indent=2, ensure_ascii=False)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Count videos
    video_count = sum(1 for p in posts if p.get('extracted_media', {}).get('media_type') == 'video')
    
    print("\n" + "=" * 70)
    print("✅ Processing Complete")
    print("=" * 70)
    print(f"📊 Processed with LLM: {len(processed_ids)}")
    print(f"⏭️  Already processed: {len(already_processed_ids)}")
    print(f"⏭️  No processing needed: {len(no_processing_needed_ids)}")
    print(f"❌ Errors: {len(error_ids)}")
    print(f"🖼️  Image posts found: {image_posts}")
    print(f"🎥 Video posts found: {video_count}")
    print(f"⏱️  Duration: {duration:.1f}s")
    print(f"💾 Updated: {input_file}")
    
    return {
        "processed": processed_ids,
        "already_processed": already_processed_ids,
        "no_processing_needed": no_processing_needed_ids,
        "error": error_ids,
        "images": image_posts,
        "videos": video_count,
        "duration": duration,
        "file": input_file
    }


def get_next_preprocess_run_number(folder: str, subreddit: str) -> int:
    """Get the next available preprocessing run number for a subreddit.
    
    Looks for existing metadata files like 'subreddit_preprocess_metadata_001.json'.
    Returns 1 if no files exist, otherwise returns max + 1.
    """
    path = Path(folder)
    if not path.exists():
        return 1
    
    # Find all preprocess_metadata files for this subreddit
    pattern = f"{subreddit}_preprocess_metadata_*.json"
    existing_files = list(path.glob(pattern))
    
    max_num = 0
    for file in existing_files:
        # Extract number from filename (e.g., "samharris_preprocess_metadata_001.json" -> 1)
        filename = file.stem  # Remove .json
        parts = filename.split('_')
        
        if parts[-1].isdigit():
            num = int(parts[-1])
            max_num = max(max_num, num)
    
    return max_num + 1 if max_num > 0 else 1


def save_preprocess_metadata(
    folder: str,
    subreddit: str,
    start_time: str,
    end_time: str,
    processed: list,
    already_processed: list,
    no_processing_needed: list,
    error: list,
    function_parameters: dict = None,
) -> str:
    """Save metadata about preprocessing run.
    
    Args:
        folder: Path to folder containing processed files
        subreddit: Subreddit name
        start_time: ISO format start time
        end_time: ISO format end time
        processed: List of post IDs successfully processed with LLM
        already_processed: List of post IDs already processed (extracted_media exists)
        no_processing_needed: List of post IDs that required no processing
        error: List of post IDs where an error occurred
        function_parameters: Dict of function parameters used (folder_path, post_ids, force_reprocess)
    
    Returns:
        str: Path to metadata file
    """
    # Get run number
    run_number = get_next_preprocess_run_number(folder, subreddit)
    
    # Create metadata
    metadata = {
        "subreddit": subreddit,
        "start_time": start_time,
        "end_time": end_time,
    }
    
    # Add function parameters if provided
    if function_parameters:
        metadata["function_parameters"] = function_parameters
    
    # Add processing results
    metadata.update({
        "processed": processed,
        "already_processed": already_processed,
        "no_processing_needed": no_processing_needed,
        "error": error,
    })
    
    # Save metadata file
    metadata_file = f"{folder}/{subreddit}_preprocess_metadata_{run_number:03d}.json"
    
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def process_folder(folder_path: str, post_ids: list = None, force_reprocess: bool = False) -> Dict:
    """Process all JSON files in a folder to extract media information.
    
    Main orchestrator function that processes all subreddit files in a folder
    and saves a preprocessing metadata file with run information.
    
    Args:
        folder_path: Path to folder containing raw data files (e.g., 'data/raw/scrape_001')
        post_ids: Optional list of post IDs to filter (applies to all files)
        force_reprocess: If True, reprocess posts even if extracted_media exists
    
    Returns:
        dict: Summary statistics about all processed files
    """
    start_time_overall = datetime.now().isoformat()
    
    folder = Path(folder_path)
    
    if not folder.exists():
        return {"error": f"Folder not found: {folder_path}"}
    
    # Find all JSON files in folder (excluding metadata files)
    json_files = [f for f in folder.glob("*.json") if "_metadata.json" not in f.name and "_preprocess_metadata_" not in f.name]
    
    if not json_files:
        return {"error": f"No JSON files found in {folder_path}"}
    
    print("=" * 70)
    print(f"ChamberCheck - Batch Media Preprocessing")
    print("=" * 70)
    print(f"Folder: {folder_path}")
    print(f"Files to process: {len(json_files)}\n")
    
    all_results = []
    
    for data_file in sorted(json_files):
        result = process_posts(str(data_file), post_ids=post_ids, force_reprocess=force_reprocess)
        all_results.append(result)
    
    end_time_overall = datetime.now().isoformat()
    
    # Save preprocessing metadata for each subreddit (only its own posts)
    metadata_files = []
    for data_file, result in zip(sorted(json_files), all_results):
        subreddit = data_file.stem  # Get filename without extension
        if "processed" in result:
            processed_ids = result.get("processed", [])
            already_processed_ids = result.get("already_processed", [])
            no_processing_needed_ids = result.get("no_processing_needed", [])
            error_ids = result.get("error", [])
        else:
            processed_ids = []
            already_processed_ids = []
            no_processing_needed_ids = []
            error_ids = []

        metadata_file = save_preprocess_metadata(
            str(folder),
            subreddit,
            start_time_overall,
            end_time_overall,
            processed_ids,
            already_processed_ids,
            no_processing_needed_ids,
            error_ids,
            function_parameters={
                "folder_path": folder_path,
                "post_ids": post_ids,
                "force_reprocess": force_reprocess
            }
        )
        metadata_files.append(metadata_file)
    
    print("\n" + "=" * 70)
    print("✅ Batch Processing Complete")
    print("=" * 70)
    print(f"Files processed: {len(all_results)}")
    total_processed = sum(len(r.get("processed", [])) for r in all_results if "processed" in r)
    total_already_processed = sum(len(r.get("already_processed", [])) for r in all_results if "processed" in r)
    total_no_processing_needed = sum(len(r.get("no_processing_needed", [])) for r in all_results if "processed" in r)
    total_error = sum(len(r.get("error", [])) for r in all_results if "processed" in r)

    print(f"Processed with LLM: {total_processed}")
    print(f"Already processed: {total_already_processed}")
    print(f"No processing needed: {total_no_processing_needed}")
    print(f"Errors: {total_error}")
    if metadata_files:
        print(f"Metadata saved: {metadata_files[0]}\n")
    
    return {
        "files_processed": len(all_results),
        "results": all_results,
        "metadata_files": metadata_files,
        "total_processed": total_processed,
        "total_already_processed": total_already_processed,
        "total_no_processing_needed": total_no_processing_needed,
        "total_error": total_error
    }


def enrich_posts_context(
    scrape_dir,
    config_path: str = "config/config.yaml",
    force: bool = False,
) -> str:
    """Enrich post-level data with media analysis and write posts_context_NNN.json.

    Reads the latest ``comments/comments_filtered_NNN.json`` from *scrape_dir*,
    processes each post one at a time, and writes every result immediately to a
    ``.jsonl`` progress file so no work is lost if the run is interrupted.

    On completion the ``.jsonl`` is promoted to:
        ``<scrape_dir>/comments/posts_context_NNN.json``
        ``<scrape_dir>/comments/posts_context_NNN_metadata.json``

    LLM is called only when there is something useful to extract:
    - **Images** (i.redd.it, direct image URLs, Reddit embedded): OpenAI Vision
      extracts visible text, description, topic — useful for screenshots of
      tweets, news articles, quotes etc.
    - **External links**: page is fetched and summarised via OpenAI text — adds
      article context to the prompt.

    No LLM call is made for:
    - Self / text-only posts (nothing visual to extract)
    - Video links (v.redd.it, YouTube etc.) — URL is recorded but not analysed
    - Already-processed posts (``extracted_media`` present) unless *force=True*

    Args:
        scrape_dir:  Path to the scrape folder (e.g. ``"data/raw/scrape_006"``).
        config_path: Path to the unified YAML config file (unused; kept for
                     signature consistency).
        force:       Re-process posts even when ``extracted_media`` already exists.

    Returns:
        Path to the written ``posts_context_NNN.json`` file as a string.
    """
    from datetime import datetime

    scrape_path = Path(scrape_dir)
    comments_dir = scrape_path / "comments"

    # --- locate latest filtered comments file ---
    candidates = sorted(
        p for p in comments_dir.glob("comments_filtered_*.json")
        if not p.name.endswith("_metadata.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No comments_filtered_*.json found in {comments_dir}"
        )
    filtered_file = candidates[-1]

    # --- auto-increment run number (from existing .json, not .jsonl) ---
    existing_json = [
        p for p in comments_dir.glob("posts_context_*.json")
        if not p.name.endswith("_metadata.json") and p.suffix == ".json"
    ]
    nums = []
    for p in existing_json:
        try:
            nums.append(int(p.stem.split("posts_context_")[-1]))
        except ValueError:
            pass

    # --- resume detection: look for an incomplete .jsonl ---
    done_ids: dict = {}  # post_id -> record (already saved)
    jsonl_path: Optional[Path] = None
    run_num: int

    incomplete_jsonl = sorted(comments_dir.glob("posts_context_*.jsonl"))
    if incomplete_jsonl:
        candidate_jsonl = incomplete_jsonl[-1]
        try:
            candidate_run = int(candidate_jsonl.stem.split("posts_context_")[-1])
        except ValueError:
            candidate_run = None

        if candidate_run is not None:
            matching_json = comments_dir / f"posts_context_{candidate_run:03d}.json"
            if not matching_json.exists():
                # This is a genuine incomplete run
                for line in candidate_jsonl.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            rec = json.loads(line)
                            if rec.get("post_id"):
                                done_ids[rec["post_id"]] = rec
                        except Exception:
                            pass
                pct = len(done_ids) / max(len(done_ids), 1) * 100
                print(
                    f"\nIncomplete run {candidate_run:03d} found: "
                    f"{len(done_ids)} posts already saved."
                )
                answer = input("Continue this run? [Y/n]: ").strip().lower()
                if answer in ("", "y", "yes"):
                    run_num = candidate_run
                    jsonl_path = candidate_jsonl
                    print(f"Resuming run {run_num:03d} — skipping {len(done_ids)} posts\n")
                else:
                    done_ids = {}
                    run_num = max(nums, default=0) + 1
                    jsonl_path = comments_dir / f"posts_context_{run_num:03d}.jsonl"
                    print(f"Starting fresh run {run_num:03d}\n")
            else:
                run_num = max(nums, default=0) + 1
                jsonl_path = comments_dir / f"posts_context_{run_num:03d}.jsonl"
        else:
            run_num = max(nums, default=0) + 1
            jsonl_path = comments_dir / f"posts_context_{run_num:03d}.jsonl"
    else:
        run_num = max(nums, default=0) + 1
        jsonl_path = comments_dir / f"posts_context_{run_num:03d}.jsonl"

    out_path = comments_dir / f"posts_context_{run_num:03d}.json"
    meta_path = comments_dir / f"posts_context_{run_num:03d}_metadata.json"

    print(f"Source      : {filtered_file}")
    print(f"Progress    : {jsonl_path}")
    print(f"Output      : {out_path}")
    print(f"Force       : {force}\n")

    # --- load posts, strip comments ---
    raw_data = json.loads(filtered_file.read_text(encoding="utf-8"))
    all_posts = raw_data if isinstance(raw_data, list) else raw_data.get("posts", [])
    posts_only = [{k: v for k, v in p.items() if k != "comments"} for p in all_posts]
    total = len(posts_only)

    # --- enrich with raw posts.json metadata (is_self, external_url, etc.) ---
    # The filtered file loses these fields during preprocessing; pull them back
    # from the original posts.json so we never need a Reddit API call just to
    # discover a post type.
    raw_posts_file = scrape_path / "posts.json"
    raw_meta_map: dict = {}
    if raw_posts_file.exists():
        try:
            rp_data = json.loads(raw_posts_file.read_text(encoding="utf-8"))
            rp_list = rp_data if isinstance(rp_data, list) else rp_data.get("posts", [])
            for rp in rp_list:
                pid = rp.get("post_id")
                if pid and rp.get("metadata"):
                    raw_meta_map[pid] = rp["metadata"]
            if raw_meta_map:
                print(f"Loaded metadata for {len(raw_meta_map)} posts from posts.json")
        except Exception as exc:
            print(f"⚠️  Could not load posts.json metadata: {exc}")

    for p in posts_only:
        pid = p.get("post_id")
        if pid and pid in raw_meta_map and not p.get("metadata"):
            p["metadata"] = raw_meta_map[pid]

    print(f"Posts to enrich: {total}  (already done: {len(done_ids)})\n")

    # --- initialise LLM provider ---
    cfg_obj = Config(config_path)
    api_key = cfg_obj.get("llm.api_key") or os.getenv("OPENAI_API_KEY")
    llm_provider = None
    if api_key:
        llm_provider = OpenAIProvider(api_key=api_key, model="gpt-4o-mini")
        print("✓ OpenAI API key found\n")
    else:
        print("ℹ️  No OpenAI API key — image/link LLM analysis will be skipped\n")

    # --- counters ---
    n_llm = 0
    n_skip = 0
    n_video = 0
    n_error = 0
    start_time = datetime.now()

    with open(jsonl_path, "a", encoding="utf-8") as progress_fh:
        for idx, post in enumerate(posts_only, start=1):
            post_id = post.get("post_id", "UNKNOWN")
            title = (post.get("title") or "")[:60]

            # Skip already-done posts (resume)
            if post_id in done_ids and not force:
                continue

            print(f"[{idx}/{total}] {title}...")

            # --- already processed in the source file ---
            if not force and isinstance(post.get("extracted_media"), dict) and post["extracted_media"]:
                record = {"post_id": post_id, "status": "already_processed",
                          "extracted_media": post["extracted_media"]}
                print("  ✓ Already processed (skipping)")
                n_skip += 1
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()
                continue

            # --- self / text post: no media to extract ---
            if post.get("metadata", {}).get("is_self"):
                record = {"post_id": post_id, "status": "no_processing_needed",
                          "extracted_media": {"status": "no_processing_needed",
                                              "reason": "self_post",
                                              "processed_at": datetime.now().isoformat()}}
                print("  ✓ Text post — no media")
                n_skip += 1
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()
                continue

            # --- classify URL ---
            image_url: Optional[str] = None
            video_url: Optional[str] = None
            link_url: Optional[str] = None

            external_url = (post.get("metadata") or {}).get("external_url", "") or ""
            external_url = external_url.strip()

            # Normalise: relative /r/ paths are self-referencing
            if external_url.startswith("/r/"):
                external_url = ""

            # Strip obvious self-referencing reddit.com permalink URLs —
            # these appear on self posts and crossposts and carry no media.
            # reddit.com/gallery/ links ARE useful → keep them.
            if external_url:
                lower_ext = external_url.lower()
                permalink = (post.get("metadata") or {}).get("permalink", "")
                is_permalink = (
                    (permalink and external_url.rstrip("/").endswith(permalink.rstrip("/")))
                    or ("reddit.com/r/" in lower_ext and "/comments/" in lower_ext
                        and post_id in lower_ext)
                )
                if is_permalink:
                    external_url = ""

            if external_url:
                lower = external_url.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) or "i.redd.it" in lower:
                    image_url = external_url
                elif lower.endswith((".mp4", ".webm", ".mov")) or "v.redd.it" in lower:
                    video_url = external_url
                elif "reddit.com/gallery/" in lower:
                    # Gallery — needs Reddit API to resolve individual image URLs
                    pass  # fall through to Reddit API call below
                elif "reddit.com" not in lower:
                    link_url = external_url
                # else: some other reddit.com URL we can't use — leave all None

            # Reddit API fallback — only for gallery posts (we already have all
            # other data from the injected metadata, so no need for a live call
            # on text posts, link posts with known external URLs etc.)
            if not image_url and not video_url and not link_url:
                ext_lower = external_url.lower() if external_url else ""
                if "reddit.com/gallery/" in ext_lower:
                    print("  🔍 Fetching gallery post details from Reddit...")
                    post_details = get_reddit_post_details(post_id)
                    if post_details:
                        image_url = extract_image_url(post_details)
                        if not image_url:
                            video_url = extract_video_url(post_details)

            # --- nothing found ---
            if not image_url and not video_url and not link_url:
                record = {"post_id": post_id, "status": "no_processing_needed",
                          "extracted_media": {"status": "no_processing_needed",
                                              "reason": "no_media_found",
                                              "processed_at": datetime.now().isoformat()}}
                print("  ℹ️  No media found")
                n_skip += 1
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()
                continue

            # --- video: record URL, no LLM ---
            if video_url:
                record = {"post_id": post_id, "status": "video",
                          "extracted_media": {"media_type": "video",
                                              "video_url": video_url,
                                              "processed_at": datetime.now().isoformat()}}
                print(f"  🎥 Video URL recorded (no LLM)")
                n_video += 1
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()
                continue

            # --- image: vision LLM ---
            if image_url:
                print(f"  🖼️  Image: {image_url[:70]}...")
                extracted_media: dict = {"media_type": "image", "image_url": image_url,
                                         "processed_at": datetime.now().isoformat()}
                if llm_provider:
                    print("  🤖 Calling vision LLM...")
                    analysis = analyze_image_with_llm(image_url, llm_provider)
                    if isinstance(analysis, dict) and "error" not in analysis:
                        # Summarise long text content
                        text_content = analysis.get("text_content")
                        if isinstance(text_content, str) and len(text_content.split()) > 300:
                            print("  📝 Summarising long text...")
                            analysis["text_content"] = summarize_content(text_content, llm_provider)
                            analysis["is_summarized"] = True
                        extracted_media["analysis"] = analysis
                        chars = len(analysis.get("text_content") or "")
                        print(f"  ✓ Done  ({chars} chars extracted)")
                        n_llm += 1
                    else:
                        err = analysis.get("error") if isinstance(analysis, dict) else str(analysis)
                        print(f"  ⚠️  Vision error: {err}")
                        extracted_media["error"] = err
                        n_error += 1
                else:
                    print("  ⚠️  No LLM provider — URL recorded only")
                    n_skip += 1
                record = {"post_id": post_id, "status": "processed" if llm_provider else "no_llm",
                          "extracted_media": extracted_media}
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()
                continue

            # --- external link: fetch + summarise ---
            if link_url:
                print(f"  🔗 Link: {link_url[:70]}...")
                extracted_media = {"media_type": "link", "link_url": link_url,
                                   "processed_at": datetime.now().isoformat()}
                print("  🔍 Fetching link content...")
                link_content = fetch_and_extract_link_content(link_url)
                if link_content:
                    if llm_provider:
                        print("  📝 Summarising...")
                        summary = summarize_content(link_content, llm_provider)
                        try:
                            topic_label = llm_provider.analyze_with_text(
                                "Extract a concise 2-6 word topic label. Return ONLY the label.\n\n"
                                f"Text:\n{summary}"
                            )
                            if isinstance(topic_label, str):
                                topic_label = " ".join(topic_label.strip().strip('"\'').split()[:6])
                        except Exception:
                            topic_label = None
                        extracted_media["extracted_text"] = summary
                        extracted_media["topic"] = topic_label
                        extracted_media["is_summarized"] = len(link_content.split()) > 300
                        print(f"  ✓ Done  ({len(summary)} chars)")
                        n_llm += 1
                    else:
                        extracted_media["extracted_text"] = link_content[:MEDIA_FALLBACK_TEXT_LENGTH]
                        n_skip += 1
                    record = {"post_id": post_id, "status": "processed",
                              "extracted_media": extracted_media}
                else:
                    print("  ⚠️  Could not fetch link")
                    extracted_media["error"] = "link_content_unavailable"
                    record = {"post_id": post_id, "status": "error",
                              "extracted_media": extracted_media}
                    n_error += 1
                progress_fh.write(json.dumps(record, default=str) + "\n")
                progress_fh.flush()

    # --- promote .jsonl → .json ---
    all_records: List[Dict] = list(done_ids.values())  # from resume
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    # Avoid duplicates when resuming
                    if rec.get("post_id") not in {r["post_id"] for r in all_records}:
                        all_records.append(rec)
                except Exception:
                    pass

    out_path.write_text(
        json.dumps({"posts": all_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    completed_at = datetime.now()
    metadata = {
        "run": run_num,
        "generated_at": start_time.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": round((completed_at - start_time).total_seconds(), 1),
        "scrape_dir": str(scrape_path),
        "source_file": str(filtered_file),
        "output_file": str(out_path),
        "force": force,
        "counts": {
            "posts_total": total,
            "processed_with_llm": n_llm,
            "video_only": n_video,
            "skipped_no_media_or_self": n_skip,
            "errors": n_error,
        },
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Remove progress file now that final JSON is written
    jsonl_path.unlink(missing_ok=True)

    print(f"\n{'='*60}")
    print(f"✅  Done")
    print(f"   LLM calls   : {n_llm}")
    print(f"   Videos      : {n_video}")
    print(f"   Skipped     : {n_skip}")
    print(f"   Errors      : {n_error}")
    print(f"   Output      : {out_path}")
    print(f"   Metadata    : {meta_path}")
    return str(out_path)

