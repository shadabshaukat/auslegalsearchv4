# Install AWS CLI v2 on Ubuntu (Official Method)

1. **Download and install unzip if missing:**
```sh
sudo apt update
sudo apt install unzip -y
```
2. **Download the AWS CLI v2 installer:**
```sh
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
```
3. **Unzip the installer:**
```sh
unzip awscliv2.zip
```
4. **Run the install script:**
```sh
sudo ./aws/install
```
5. **Verify the install:**
```sh
aws --version
```
6. **(Optional) Clean up:**
```sh
rm -rf awscliv2.zip aws/
```
7. **Configure:**  
```sh
aws configure
```
And follow the prompts (enter your credentials).

---

*Reference: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html*
