"""
Note: https://www.learnaws.org/2020/12/18/aws-ses-boto3-guide/
"""

from pathlib import Path

from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import boto3


class AmazonSES(object):

    def __init__(self, region, access_key, secret_key, from_address, charset="UTF-8"):
        self.region = region
        self.access_key = access_key
        self.secret_key = secret_key
        self.client = boto3.client(
            "ses",
            region_name=self.region,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
        )
        self.CHARSET = charset
        self.from_address = from_address

    def send_text_email(self, to_address, subject, content):

        response = self.client.send_email(
            Destination={
                "ToAddresses": [to_address],
            },
            Message={
                "Body": {
                    "Text": {
                        "Charset": self.CHARSET,
                        "Data": content,
                    }
                },
                "Subject": {
                    "Charset": self.CHARSET,
                    "Data": subject,
                },
            },
            Source=self.from_address,
        )

    def send_html_email(self, to_address, subject, content):
        response = self.client.send_email(
            Destination={
                "ToAddresses": [
                    to_address,
                ],
            },
            Message={
                "Body": {
                    "Html": {
                        "Charset": self.CHARSET,
                        "Data": content,
                    }
                },
                "Subject": {
                    "Charset": self.CHARSET,
                    "Data": subject,
                },
            },
            Source=self.from_address,
        )

    def send_html_email_many(self, to_addresses, subject, content):
        # Accepts list[str] or tuple[str,...]
        for to in to_addresses:
            self.send_html_email(to, subject, content)

    def send_html_email_with_inline_images(
        self,
        to_address,
        subject,
        html_content,
        inline_images=None,
        image_paths=None,
    ):
        """
        Send HTML email with images embedded as MIME related parts and referenced
        from the HTML body using cid:<content_id>.
        """
        inline_images = self._normalize_inline_images(
            inline_images=inline_images, image_paths=image_paths
        )

        msg_root = MIMEMultipart("related")
        msg_root["Subject"] = subject
        msg_root["From"] = self.from_address
        msg_root["To"] = to_address

        msg_alternative = MIMEMultipart("alternative")
        msg_root.attach(msg_alternative)
        msg_alternative.attach(MIMEText(html_content, "html", self.CHARSET))

        for image in inline_images:
            img = MIMEImage(image["data"], _subtype=image.get("subtype", "png"))
            img.add_header("Content-ID", f"<{image['content_id']}>")
            img.add_header("Content-Disposition", "inline", filename=image["filename"])
            msg_root.attach(img)

        self.client.send_raw_email(
            Source=self.from_address,
            Destinations=[to_address],
            RawMessage={"Data": msg_root.as_string()},
        )

    def send_html_email_many_with_inline_images(
        self,
        to_addresses,
        subject,
        html_content,
        inline_images=None,
        image_paths=None,
    ):
        for to in to_addresses:
            self.send_html_email_with_inline_images(
                to,
                subject,
                html_content,
                inline_images=inline_images,
                image_paths=image_paths,
            )

    def _normalize_inline_images(self, inline_images=None, image_paths=None):
        if inline_images is not None and image_paths is not None:
            raise ValueError("Pass either inline_images or image_paths, not both.")

        if inline_images is not None:
            return inline_images

        if image_paths is None:
            return []

        images = []
        for image_path in image_paths:
            path = Path(image_path)
            subtype = path.suffix.lower().lstrip(".") or "png"
            images.append(
                {
                    "data": path.read_bytes(),
                    "subtype": subtype,
                    "content_id": path.name,
                    "filename": path.name,
                }
            )
        return images
