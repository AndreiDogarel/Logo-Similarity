import os
import requests
import pyarrow.parquet as pq

class FaviconDownloader:
    GOOGLE_FAVICON_API = "https://www.google.com/s2/favicons?sz=64&domain={}"
    
    def __init__(self, parquet_file, output_dir="favicons"):
        self.parquet_file = parquet_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_domains(self):
        try:
            table = pq.read_table(self.parquet_file)
            df = table.to_pandas()
            if "domain" not in df.columns:
                raise ValueError("No domains found")
            return df["domain"].dropna().unique()
        except Exception as e:
            print(f"Error while reading Parquet file {e}")
            return []
    
    def download_favicon(self, domain):
        url = self.GOOGLE_FAVICON_API.format(domain)
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, f"{domain}.png")
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Saved: {filepath}")
            else:
                print(f"Error ({response.status_code}) for {domain}")
        except Exception as e:
            print(f"Error while downloading icon for {domain}: {e}")
    
    def run(self):
        domains = self.get_domains()
        for domain in domains:
            self.download_favicon(domain)
