#!/bin/bash
set -euo pipefail
source .env.cloudbuild

gcloud config configurations activate default
gcloud config set project "$PROJECT_ID"
gcloud auth application-default set-quota-project "$PROJECT_ID"

PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
GITHUB_USERNAME=$(gh api user --jq '.login')
GITHUB_REPO_NAME=$(basename "$(git remote get-url origin)" .git)
GITHUB_REPO_FULL=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')
SA=$(echo "$GITHUB_REPO_NAME-actions" | sed 's/_/-/g')
SA_EMAIL="$SA@${PROJECT_ID}.iam.gserviceaccount.com"
WORKLOAD_IDENTITY_PROVIDER="projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/providers/$WIF_PROVIDER"
WIF_PRINCIPAL_SET="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$WIF_POOL/attribute.repository/$GITHUB_REPO_FULL"


echo "GCP Project: $PROJECT_ID"
echo "REPO=$GITHUB_REPO_FULL"

grep -v "WORKLOAD_IDENTITY_PROVIDER\|SA_EMAIL" .env.cloudbuild > .env.cloudbuild.swp
echo "SA_EMAIL=$SA_EMAIL" | tee -a .env.cloudbuild.swp
echo "WORKLOAD_IDENTITY_PROVIDER=$WORKLOAD_IDENTITY_PROVIDER" | tee -a .env.cloudbuild.swp
mv .env.cloudbuild.swp .env.cloudbuild

# Enable IAM Credentials API
gcloud services list > /tmp/services_enabled.txt

if ! grep -q "iamcredentials.googleapis.com" /tmp/services_enabled.txt; then
    gcloud services enable iamcredentials.googleapis.com
fi

if ! grep -q "cloudresourcemanager.googleapis.com" /tmp/services_enabled.txt; then
    gcloud services enable cloudresourcemanager.googleapis.com
fi

sleep 60 # wait for services to be enabled

# Create GCS Bucket
if gcloud storage buckets list | grep -q "$GCS_BUCKET"; then
    echo "Bucket $GCS_BUCKET already exists."
else
    gcloud storage buckets create "gs://$GCS_BUCKET" \
      --project="$PROJECT_ID" \
      --public-access-prevention \
      --uniform-bucket-level-access \
      --location="$GCS_REGION"

    echo "Bucket $GCS_BUCKET created."
fi

# Create Workload Identity Pool
if gcloud iam workload-identity-pools list --location="global" | grep -q "$WIF_POOL"; then
    echo "Workload Identity Pool $WIF_POOL already exists."
else
    gcloud iam workload-identity-pools create "$WIF_POOL" \
      --project="${PROJECT_ID}" \
      --location="global" \
      --display-name="GitHub Actions Pool"

    echo "Workload Identity Pool $WIF_POOL created."
fi

# Create Workload Identity Pool Provider
if gcloud iam workload-identity-pools providers list --workload-identity-pool="$WIF_POOL" --location="global" | grep -q "$WIF_PROVIDER"; then
    echo "Workload Identity Pool Provider $WIF_PROVIDER already exists."
else
    gcloud iam workload-identity-pools providers create-oidc "$WIF_PROVIDER" \
      --project="${PROJECT_ID}" \
      --location="global" \
      --workload-identity-pool="$WIF_POOL" \
      --display-name="GitHub OIDC Provider" \
      --issuer-uri="https://token.actions.githubusercontent.com" \
      --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
      --attribute-condition="assertion.repository_owner == '$GITHUB_USERNAME'"

    echo "Workload Identity Pool Provider $WIF_PROVIDER created. Provider for all GitHub repositories owned by $GITHUB_USERNAME."
fi

# Create Service Account
if gcloud iam service-accounts list --project="${PROJECT_ID}" | grep -q "$SA"; then
    echo "Service Account $SA already exists."
else
    gcloud iam service-accounts create "$SA" \
      --project="${PROJECT_ID}" \
      --display-name="$GITHUB_REPO_FULL Actions SA"
fi

# Grant the Actions service account access to the bucket (idempotent, always run)
gcloud storage buckets add-iam-policy-binding "gs://$GCS_BUCKET" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/storage.objectAdmin"
echo "Bucket IAM policy binding ensured for $SA_EMAIL."

# Add IAM policy binding for Workload Identity Pool (idempotent, always run)
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="${WIF_PRINCIPAL_SET}"
echo "Workload Identity IAM policy binding ensured for $WIF_PRINCIPAL_SET."
