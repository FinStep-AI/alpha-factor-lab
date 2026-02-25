#!/usr/bin/env python3
"""
飞书 Interactive Card 发送工具
用法: python3 feishu_send_card.py --chat <chat_id> --card <card.json>
      python3 feishu_send_card.py --chat <chat_id> --card-str '<json_string>'

从 openclaw.json 自动读取飞书 appId/appSecret。
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


def get_feishu_credentials():
    """从 openclaw.json 读取飞书凭证"""
    config_path = os.path.expanduser("~/.openclaw/openclaw.json")
    with open(config_path) as f:
        cfg = json.load(f)
    feishu = cfg.get("channels", {}).get("feishu", {})
    accounts = feishu.get("accounts", {})
    main = accounts.get("main", {})
    app_id = main.get("appId") or feishu.get("appId")
    app_secret = main.get("appSecret") or feishu.get("appSecret")
    if not app_id or not app_secret:
        raise ValueError("Missing feishu appId/appSecret in openclaw.json")
    return app_id, app_secret


def get_tenant_access_token(app_id, app_secret):
    """获取 tenant_access_token"""
    url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
    data = json.dumps({"app_id": app_id, "app_secret": app_secret}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
    if result.get("code") != 0:
        raise ValueError(f"Failed to get token: {result}")
    return result["tenant_access_token"]


def send_card(token, chat_id, card):
    """发送 interactive card 到指定 chat"""
    url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    payload = {
        "receive_id": chat_id,
        "msg_type": "interactive",
        "content": json.dumps(card, ensure_ascii=False)
    }
    data = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {token}"
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        result = json.loads(resp.read())
    if result.get("code") != 0:
        raise ValueError(f"Send failed: {result}")
    msg_id = result.get("data", {}).get("message_id", "?")
    print(f"✅ Card sent! message_id={msg_id}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Send Feishu interactive card")
    parser.add_argument("--chat", required=True, help="Chat ID (oc_xxx)")
    parser.add_argument("--card", help="Path to card JSON file")
    parser.add_argument("--card-str", help="Card JSON as string")
    args = parser.parse_args()

    if args.card:
        with open(args.card) as f:
            card = json.load(f)
    elif args.card_str:
        card = json.loads(args.card_str)
    else:
        # Read from stdin
        card = json.load(sys.stdin)

    app_id, app_secret = get_feishu_credentials()
    token = get_tenant_access_token(app_id, app_secret)
    send_card(token, args.chat, card)


if __name__ == "__main__":
    main()
