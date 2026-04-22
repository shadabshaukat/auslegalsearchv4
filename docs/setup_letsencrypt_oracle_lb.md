# Enable HTTPS via Let's Encrypt for Streamlit behind an OCI Load Balancer

## Architecture

- Public DNS: auslegal.oraclecloudserver.com (A record: 64.181.241.85 → OCI LB)
- OCI Load Balancer, HTTP frontend on port 80, route to backend set (private IP, HTTP:8501)
- Streamlit running on private VM on 10.150.1.82:8501 (no public access)
- Objective: SSL termination and HTTPS on auslegal.oraclecloudserver.com

## RECOMMENDED: SSL TERMINATION ON OCI LOAD BALANCER

### Steps

1. **Obtain SSL Certificate from Let's Encrypt (On Any Linux VM)**

You'll need a valid certificate for auslegal.oraclecloudserver.com. Use DNS challenge if your LB does not support direct HTTP challenge to backend.
- Install Certbot and DNS plugin for Route53:

```sh
sudo apt update
sudo apt install certbot python3-certbot-dns-route53
```
2. **Request Certificate with DNS Challenge (Requires AWS CLI creds for Route53 zone):**

```sh
sudo certbot certonly \
  --dns-route53 \
  -d auslegal.oraclecloudserver.com \
  -d www.auslegal.oraclecloudserver.com
```
Certs save to: `/etc/letsencrypt/live/auslegal.oraclecloudserver.com/`

3. **Upload SSL Certificate to OCI Load Balancer (as SSL/HTTPS Listener Certificate)**

- You'll need:
  - `fullchain.pem` (certificate with intermediates)
  - `privkey.pem` (private key)
  - (possibly chain pem, if not using fullchain)
- In OCI Console: **Load Balancers → Your LB → Management → Certificates**  
  - Create/Import Certificate
  - Upload those PEM files as prompted or Paste the contents of the .pem files
- **Important Note**
  - SSL Certificate in OCI is `fullchain.pem`
  - CA Certificate in OCI is `cert.pem`
  - Private Key in OCI is `privkey.pem`   
   
4. **Configure Listener and Rule for HTTPS**

- In LB, create a new listener (HTTPS:443) using the certificate.
- Forward/proxy to the backend set on port 8501.
- (Optionally, redirect port 80 HTTP traffic to HTTPS in LB rules).

5. **Automate Renewal**

- Let's Encrypt certs expire in 90 days. To automate:
  - Run `sudo certbot renew` regularly (e.g., cron or systemd timer).
  - After renewal, script upload to OCI (can use oci-cli).
  - [OCI CLI docs for certificate import:](https://docs.oracle.com/en-us/iaas/tools/oci-cli/3.31.2/oci_cli_docs/cmdref/lb/certificate/create.html)
  - Example (requires OCI CLI configured with proper user/API keys):

```sh
oci lb certificate update \
  --load-balancer-id <snippet from your OCI console> \
  --certificate-name "my-letsencrypt-cert" \
  --private-key file://privkey.pem \
  --public-certificate file://fullchain.pem
```
Automate this upload after every renewal.

---

## Why Is This Best?

- *Secure*: Backend never exposed to the public.
- *Reliable*: Certbot/LB management is completely decoupled from app code/process.
- *Automatable*: Use DNS challenge & scripted OCI CLI upload for seamless renewal.

---

**Your public users see HTTPS to your domain, handled entirely by the OCI Load Balancer. Your Streamlit server remains private and only needs HTTP.**
