#!/usr/bin/env python3
import os
from dotenv import load_dotenv
import sib_api_v3_sdk
from sib_api_v3_sdk import Configuration, ApiClient, TransactionalEmailsApi, SendSmtpEmail
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')

# Load environment variables
logger.info(f"Looking for .env file at: {env_path}")
if os.path.exists(env_path):
    logger.info("Found .env file")
    load_dotenv(env_path)
else:
    logger.error("❌ .env file not found!")

def test_brevo_connection():
    """Test the Brevo API connection and credentials"""
    try:
        # Get API key from environment
        api_key = os.getenv("BREVO_API_KEY")
        if not api_key:
            logger.error("❌ BREVO_API_KEY not found in environment variables")
            logger.info("Available environment variables:")
            for key in os.environ:
                if key.startswith("BREVO") or key.startswith("SIB"):
                    masked_value = f"{os.environ[key][:4]}...{os.environ[key][-4:]}"
                    logger.info(f"  {key}: {masked_value}")
            raise ValueError("BREVO_API_KEY not found in environment variables")
        
        # Log masked API key for debugging
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        logger.info(f"Using API key: {masked_key}")
        
        # Initialize Brevo client
        configuration = Configuration()
        configuration.api_key['api-key'] = api_key
        api_instance = TransactionalEmailsApi(ApiClient(configuration))
        
        # Test API connection by getting SMTP templates
        logger.info("Testing API connection...")
        
        # Try to send a test email to verify the API key
        sender = {"name": "Test", "email": "test@test.com"}
        to = [{"email": "test@test.com"}]
        subject = "API Test"
        html_content = "<html><body><p>Test</p></body></html>"
        
        send_smtp_email = SendSmtpEmail(
            to=to,
            html_content=html_content,
            sender=sender,
            subject=subject
        )
        
        # We don't actually send the email, just validate the API key
        api_instance.send_transac_email_with_http_info(send_smtp_email, _preload_content=False)
        logger.info("✅ Brevo API connection successful!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Brevo API: {str(e)}")
        if hasattr(e, 'body'):
            try:
                import json
                error_details = json.loads(e.body)
                logger.error(f"Error details: {error_details}")
            except:
                logger.error(f"Error details: {e.body}")
        return False

def send_test_email(to_email: str):
    """Send a test email using Brevo"""
    try:
        # Get API key from environment
        api_key = os.getenv("BREVO_API_KEY")
        if not api_key:
            raise ValueError("BREVO_API_KEY not found in environment variables")
        
        # Log masked API key for debugging
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        logger.info(f"Using API key: {masked_key}")
        
        # Initialize Brevo client
        configuration = Configuration()
        configuration.api_key['api-key'] = api_key
        api_instance = TransactionalEmailsApi(ApiClient(configuration))
        
        # Prepare email content
        # Using Brevo's default verified domain
        sender = {"name": "Kokoro TTS Test", "email": "jaldao27@gmail.com"}
        to = [{"email": to_email}]
        subject = "Kokoro TTS - Test Email"
        html_content = """
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h1 style="color: #2c3e50; text-align: center;">Test Email</h1>
                    <p>This is a test email from Kokoro TTS.</p>
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <p style="margin: 0;">If you're seeing this, the email system is working correctly! 🎉</p>
                    </div>
                    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="text-align: center; color: #7f8c8d; font-size: 0.8em;">
                        This is an automated test email from Kokoro TTS
                    </p>
                </div>
            </body>
        </html>
        """
        
        # Create email object
        send_smtp_email = SendSmtpEmail(
            to=to,
            html_content=html_content,
            sender=sender,
            subject=subject,
            # Add reply-to address
            reply_to={"email": "jaldao27@gmail.com", "name": "Kokoro TTS Support"}
        )
        
        # Send email
        logger.info(f"Sending test email to {to_email}...")
        logger.info("Email configuration:")
        logger.info(f"- From: {sender['name']} <{sender['email']}>")
        logger.info(f"- To: {to_email}")
        logger.info(f"- Subject: {subject}")
        
        response = api_instance.send_transac_email(send_smtp_email)
        logger.info("✅ Test email sent successfully!")
        logger.info(f"Response: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send test email: {str(e)}")
        if hasattr(e, 'body'):
            try:
                import json
                error_details = json.loads(e.body)
                logger.error(f"Error details: {error_details}")
            except:
                logger.error(f"Error details: {e.body}")
        return False

def send_template_email(to_email: str, download_url: str, template_name: str = "Audiobook"):
    """Send an email using a Brevo template"""
    try:
        # Get API key from environment
        api_key = os.getenv("BREVO_API_KEY")
        if not api_key:
            raise ValueError("BREVO_API_KEY not found in environment variables")
        
        # Initialize Brevo client
        configuration = Configuration()
        configuration.api_key['api-key'] = api_key
        api_instance = TransactionalEmailsApi(ApiClient(configuration))
        
        # Get template ID by name
        templates = api_instance.get_smtp_templates()
        template_id = None
        for template in templates.templates:
            if template.name == template_name:
                template_id = template.id
                break
        
        if template_id is None:
            logger.error(f"❌ Template '{template_name}' not found")
            logger.info("Available templates:")
            for template in templates.templates:
                logger.info(f"  - {template.name} (ID: {template.id})")
            raise ValueError(f"Template '{template_name}' not found")
        
        logger.info(f"Found template: {template_name} (ID: {template_id})")
        
        # Prepare email parameters
        sender = {"name": "Kokoro TTS", "email": "jaldao27@gmail.com"}
        to = [{"email": to_email}]
        
        # Parameters that will replace variables in the template
        params = {
            "download_url": download_url,
            "user_email": to_email
        }
        
        # Create email object using template
        send_smtp_email = SendSmtpEmail(
            to=to,
            sender=sender,
            template_id=template_id,
            params=params,
            reply_to={"email": "jaldao27@gmail.com", "name": "Kokoro TTS Support"}
        )
        
        # Send email
        logger.info(f"Sending template email to {to_email}...")
        logger.info("Email configuration:")
        logger.info(f"- Template: {template_name} (ID: {template_id})")
        logger.info(f"- From: {sender['name']} <{sender['email']}>")
        logger.info(f"- To: {to_email}")
        logger.info(f"- Parameters: {params}")
        
        response = api_instance.send_transac_email(send_smtp_email)
        logger.info("✅ Template email sent successfully!")
        logger.info(f"Response: {response}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to send template email: {str(e)}")
        if hasattr(e, 'body'):
            try:
                import json
                error_details = json.loads(e.body)
                logger.error(f"Error details: {error_details}")
            except:
                logger.error(f"Error details: {e.body}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Brevo email functionality')
    parser.add_argument('--email', type=str, help='Email address to send test email to')
    parser.add_argument('--template', action='store_true', help='Use template instead of raw email')
    parser.add_argument('--url', type=str, default='https://example.com/download/123', help='Download URL for template email')
    args = parser.parse_args()
    
        # First test the connection
        logger.info("Testing Brevo API connection...")
        if test_brevo_connection():
            if args.email:
                if args.template:
                    # Send email using template
                    send_template_email(args.email, args.url)
                else:
                    # Send regular test email
                    send_test_email(args.email)
            else:
                logger.info("No email address provided. Use --email to send a test email.")
                logger.info("Example: python test_email.py --email your@email.com")
                logger.info("For template email: python test_email.py --email your@email.com --template --url https://your-url.com")
        else:
            logger.error("Skipping email test due to connection failure.") 