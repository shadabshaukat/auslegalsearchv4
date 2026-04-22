# Certbot DNS Route53: "Unable to locate credentials" Solution

To allow Certbot (with --dns-route53) to create the DNS challenge, you must:

## 1. Create an IAM User with Route53 Permissions

- Go to AWS Console → IAM → Users → "Add User"
    - Name: certbot-route53
    - Enable "Programmatic access"

- Attach policy (JSON below) or use "AmazonRoute53FullAccess" for simplicity.
- Download or copy Access Key ID and Secret Access Key.

**Recommended minimum policy** (replace `<your-zone-id>` if you want to lock down further):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "route53:GetChange",
        "route53:ListHostedZones",
        "route53:ListResourceRecordSets",
        "route53:ChangeResourceRecordSets"
      ],
      "Resource": "*"
    }
  ]
}
```
## 2. Configure Credentials for Certbot/boto3

### Option 1: Environment Variables (Best for one-shot usage)

```sh
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
```
### Option 2: Shared AWS credentials file

- Edit/create `~/.aws/credentials`:

```
[default]
aws_access_key_id = your-access-key-id
aws_secret_access_key = your-secret-access-key
```
- (Optionally add `~/.aws/config` and set region if you like, e.g.):
```
[default]
region = us-east-1
```
## 3. Retry Certbot Command

```sh
sudo certbot certonly \
  --dns-route53 \
  -d auslegal.oraclecloudserver.com \
  -d www.auslegal.oraclecloudserver.com
```
## 4. Confirm Renewal Works

Once issued, certificate renewal (renew hook) will also require access via these credentials.

---

**Note:** Protect your credentials and remove user or rotate keys after use if possible!
