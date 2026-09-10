import sys
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def send_test(recipient_email, alert_level="DANGER"):
    smtp_host = "smtp.gmail.com"
    smtp_port = 587
    smtp_pass = os.getenv("SMTP_PASS", "")
    if not smtp_pass and os.path.exists(".env.local"):
        try:
            with open(".env.local", "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("SMTP_PASS="):
                        smtp_pass = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    if not smtp_pass:
        smtp_pass = input("Enter 16-letter Gmail App Password: ").strip()
    smtp_pass = smtp_pass.replace(" ", "").strip()
    smtp_user = os.getenv("SMTP_USER", "miningwithigniters@gmail.com")
    subject = f"[{alert_level} MINE HAZARD ALERT] Evacuate Immediately — Kolar Gold Fields"
    msg_text = "Alert: you have to move from that current site"

    print(f"\n=======================================================")
    print(f"[ALERT] INITIATING EMERGENCY HAZARD ALERT DISPATCH")
    print(f"From       : {smtp_user}")
    print(f"To         : {recipient_email}")
    print(f"Alert Level: {alert_level}")
    print(f"Message    : {msg_text}")
    print(f"=======================================================")

    plain_content = f"""EMERGENCY HAZARD NOTIFICATION
---------------------------------------------
{msg_text}

Status Level  : {alert_level}
Mine Location : Kolar Gold Fields
Sensor Node   : NODE_01
Telemetry     : Tilt: 5.25 deg/m | Vibration: 2.10 g | Strain: 3.40 mm
Timestamp     : {time.strftime('%Y-%m-%d %H:%M:%S')}

INSTRUCTION:
Hazard monitoring sensors have detected high-risk ground instability.
You have to move from that current site immediately and report to the designated safe zone.
---------------------------------------------
Enterprise Mine Subsidence Monitoring System &bull; Igniters AI
"""

    badge_bg = "#dc2626" if alert_level == "DANGER" else "#d97706"
    html_content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, sans-serif; background-color: #0b1120; color: #f1f5f9; padding: 20px; margin: 0;">
  <table width="100%" cellpadding="0" cellspacing="0" style="max-width: 600px; margin: 0 auto; background-color: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid #334155; box-shadow: 0 10px 25px rgba(0,0,0,0.5);">
    <tr>
      <td style="background-color: {badge_bg}; padding: 20px; text-align: center;">
        <h1 style="color: #ffffff; margin: 0; font-size: 22px; letter-spacing: 1px;">🚨 EMERGENCY MINE HAZARD ALERT</h1>
        <p style="color: #fef08a; margin: 6px 0 0 0; font-weight: bold; font-size: 15px;">IMMEDIATE EVACUATION NOTICE</p>
      </td>
    </tr>
    <tr>
      <td style="padding: 24px;">
        <div style="background-color: rgba(220, 38, 38, 0.15); border-left: 5px solid {badge_bg}; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
          <p style="font-size: 18px; font-weight: bold; color: #f87171; margin: 0;">{msg_text}</p>
        </div>
        <table width="100%" cellpadding="8" cellspacing="0" style="border-collapse: collapse; font-size: 14px; margin-bottom: 20px;">
          <tr style="border-bottom: 1px solid #334155;">
            <td style="color: #94a3b8;"><strong>Hazard Level:</strong></td>
            <td><span style="background-color: {badge_bg}; color: #fff; padding: 3px 10px; border-radius: 4px; font-weight: bold;">{alert_level}</span></td>
          </tr>
          <tr style="border-bottom: 1px solid #334155;">
            <td style="color: #94a3b8;"><strong>Mine Site:</strong></td>
            <td style="color: #ffffff; font-weight: bold;">Kolar Gold Fields</td>
          </tr>
          <tr style="border-bottom: 1px solid #334155;">
            <td style="color: #94a3b8;"><strong>Sensor Node:</strong></td>
            <td style="color: #38bdf8; font-weight: bold;">NODE_01</td>
          </tr>
          <tr style="border-bottom: 1px solid #334155;">
            <td style="color: #94a3b8;"><strong>Telemetry:</strong></td>
            <td style="color: #cbd5e1;">Tilt: <strong>5.25</strong> deg/m | Vibration: <strong>2.10</strong> g | Strain: <strong>3.40</strong> mm</td>
          </tr>
          <tr>
            <td style="color: #94a3b8;"><strong>Timestamp:</strong></td>
            <td style="color: #cbd5e1;">{time.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
          </tr>
        </table>
        <div style="background-color: #0f172a; padding: 14px; border-radius: 6px; text-align: center; border: 1px solid #334155;">
          <p style="color: #fbbf24; margin: 0; font-weight: bold; font-size: 14px;">⚡ Immediate Action: Evacuate current sector to surface shelter.</p>
        </div>
      </td>
    </tr>
    <tr>
      <td style="background-color: #0f172a; padding: 12px; text-align: center; font-size: 12px; color: #64748b; border-top: 1px solid #334155;">
        Automated Dispatch from Enterprise Early Warning System &bull; Igniters AI
      </td>
    </tr>
  </table>
</body>
</html>
"""

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"Igniters Mine Safety Alerts <{smtp_user}>"
        msg["To"] = recipient_email
        msg.attach(MIMEText(plain_content, "plain"))
        msg.attach(MIMEText(html_content, "html"))

        print(f"Connecting to {smtp_host}:{smtp_port} via TLS...")
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=12)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        print(f"Transmitting emergency alert message to {recipient_email}...")
        server.sendmail(smtp_user, [recipient_email], msg.as_string())
        server.quit()
        print(f"\n[SUCCESS] REAL EMAIL DELIVERED TO INBOX!")
        print(f"Recipient: {recipient_email}")
        print(f"Please check the inbox (or Spam/Junk folder if not whitelisted).")
        return True
    except Exception as e:
        print(f"\n[ERROR] Delivery failed: {e}")
        return False

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "miningwithigniters@gmail.com"
    level = sys.argv[2] if len(sys.argv) > 2 else "DANGER"
    send_test(target, level)
