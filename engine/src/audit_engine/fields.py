"""Field types shared by the engine's settings and file formats."""

import ipaddress
import re
from typing import Annotated

from pydantic import AfterValidator, HttpUrl

HEADER_SAFE = re.compile(r"[!-~]+")  # what a secret needs to travel in an HTTP header


def _check_encodable(text: str) -> str:
    text.encode()  # broken Unicode raises here, and could never be saved to a file
    return text


def _check_not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must not be blank.")
    return text


def _is_this_computer(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_encrypted(url: HttpUrl) -> HttpUrl:
    if url.scheme != "https" and not _is_this_computer((url.host or "").strip("[]")):
        raise ValueError("must use https unless it points at this computer.")
    return url


Text = Annotated[str, AfterValidator(_check_encodable), AfterValidator(_check_not_blank)]
EncryptedUrl = Annotated[HttpUrl, AfterValidator(_check_encrypted)]
