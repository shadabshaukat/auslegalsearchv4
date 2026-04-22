# Secure Your Streamlit App On Ubuntu With Let's Encrypt (Nginx Reverse Proxy)

## 1. Install Nginx & Certbot

```sh
sudo apt update
sudo apt install nginx certbot python3-certbot-nginx
```
## 2. Start Nginx

```sh
sudo systemctl enable nginx
sudo systemctl start nginx
```
## 3. Configure Nginx Server Block

- Edit or create a file for your domain:
 ```
  sudo nano /etc/nginx/sites-available/yourdomain.com
  ```
- Example config (replace yourdomain.com and change upstream port if needed):

```
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
- Enable the config:
 ```sh
  sudo ln -s /etc/nginx/sites-available/yourdomain.com /etc/nginx/sites-enabled/
  sudo nginx -t
  sudo systemctl reload nginx
  ```
## 4. Obtain a Let's Encrypt SSL Certificate

Replace `yourdomain.com` below!

```sh
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```
Follow prompts and agree to redirect HTTP to HTTPS.

## 5. Auto-Renewal of SSL

Certbot's systemd timer runs twice daily. To test:
```sh
sudo certbot renew --dry-run
```
You should see "Congratulations, all renewals succeeded!"

## 6. Reload Streamlit as Needed

Keep your Streamlit app running (e.g., with systemd or a process manager).
Nginx handles HTTPS/SSL and proxies requests to Streamlit over HTTP on localhost.

---

**Now your app is available via `https://yourdomain.com` with automated, free SSL.**
