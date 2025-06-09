import argparse
import os
import subprocess
from google.cloud import storage

DATASETS = {
    "uterine": {
        "cell_by_gene.csv": "gs://vz-ffpe-showcase/HumanUterineCancerPatient2-ROCostain/cell_by_gene.csv",
        "cell_metadata.csv": "gs://vz-ffpe-showcase/HumanUterineCancerPatient2-ROCostain/cell_metadata.csv"
    },
    "colon": {
        "cell_by_gene.csv": "gs://vz-ffpe-showcase/HumanColonCancerPatient2/cell_by_gene.csv",
        "cell_metadata.csv": "gs://vz-ffpe-showcase/HumanColonCancerPatient2/cell_metadata.csv"
    }
}

def download_with_gsutil(gcs_path, output_path):
    print(f"[INFO] Downloading {gcs_path} to {output_path} using gsutil...")
    try:
        subprocess.run(["gsutil", "cp", gcs_path, output_path], check=True)
        print(f"[✓] Downloaded {gcs_path} to {output_path}")
    except Exception as e:
        print(f"[ERROR] gsutil failed: {e}")

def download_with_gcloud_storage(gcs_path, output_path):
    print(f"[INFO] Downloading {gcs_path} to {output_path} using google-cloud-storage...")
    client = storage.Client()
    if not gcs_path.startswith("gs://"):
        raise ValueError("gcs_path must start with gs://")
    parts = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name = parts[1]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(output_path)
    print(f"[✓] Downloaded {gcs_path} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Download Vizgen MERSCOPE data from Google Cloud Storage.")
    parser.add_argument("--dataset", type=str, choices=["uterine", "colon"], help="Preset dataset to download (uterine or colon).")
    parser.add_argument("--gcs-path", type=str, help="GCS path (gs://...) to a specific file.")
    parser.add_argument("--output", type=str, help="Local output file path (used with --gcs-path).")
    parser.add_argument("--method", type=str, choices=["gsutil", "gcloud"], default="gsutil", help="Download method: gsutil or gcloud (google-cloud-storage).")
    parser.add_argument("--output-dir", type=str, default="data/merscope", help="Directory to save preset dataset files.")
    args = parser.parse_args()

    if args.dataset:
        os.makedirs(args.output_dir, exist_ok=True)
        files = DATASETS[args.dataset]
        for fname, gcs_path in files.items():
            output_path = os.path.join(args.output_dir, f"{args.dataset}_{fname}")
            if args.method == "gsutil":
                download_with_gsutil(gcs_path, output_path)
            else:
                download_with_gcloud_storage(gcs_path, output_path)
    elif args.gcs_path and args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if args.method == "gsutil":
            download_with_gsutil(args.gcs_path, args.output)
        else:
            download_with_gcloud_storage(args.gcs_path, args.output)
    else:
        print("Specify either --dataset (uterine or colon) or both --gcs-path and --output.")

if __name__ == "__main__":
    main()