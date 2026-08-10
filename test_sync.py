def process_urls_parallel(url_list, max_workers=5):
    import concurrent.futures, urllib.request
    def fetch_url(url):
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return url, resp.getcode(), len(resp.read())
        except Exception as e:
            return url, None, str(e)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(fetch_url, url) for url in url_list]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    return results
