#!/bin/bash
set -eu

GIT_REPO="https://github.com/feliceeeee/Credit_Score_Classification_AWS_Pipeline.git"
SUBFOLDER=""
APP_FILE="app_streamlit.py"
ENDPOINT_NAME="credit-score-endpoint"

REGION="us-east-1"
APP_DIR="/opt/credit-score-app"
VENV_DIR="/opt/streamlit-venv"

if [ -z "$SUBFOLDER" ]; then
    APP_PATH="$APP_DIR"
else
    APP_PATH="$APP_DIR/$SUBFOLDER"
fi

dnf update -y
dnf install -y python3 python3-pip git

git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

if [ ! -f "$APP_PATH/$APP_FILE" ]; then
    echo "Application file not found."
    find "$APP_DIR" -maxdepth 4 -type f
    exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
cd "$APP_PATH"
"$VENV_DIR/bin/pip" install -r requirements.txt

cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Credit Score Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_PATH
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=$VENV_DIR/bin/streamlit run $APP_FILE \
--server.address 0.0.0.0 \
--server.port 8501 \
--server.headless true \
--server.enableCORS false \
--server.enableXsrfProtection false
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit

sleep 5
if systemctl is-active --quiet streamlit; then
    echo "Credit Score Streamlit started successfully"
    touch "$APP_DIR/.userdata-success"
    chown ec2-user:ec2-user "$APP_DIR/.userdata-success"
else
    echo "Failed to start Streamlit"
    journalctl -u streamlit -n 50 --no-pager
    exit 1
fi
