"""
Family Contact System
Handles calling family members (mom, dad) for the bedside assistant
"""

import logging
from datetime import datetime
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FamilyContacts:
    """Manages family contact information and calling functionality"""

    def __init__(self):
        """Initialize family contacts"""
        self.contacts = {
            "mom": {
                "name": "Mom",
                "phone": None,  # To be configured
                "last_called": None,
            },
            "dad": {
                "name": "Dad",
                "phone": None,  # To be configured
                "last_called": None,
            },
        }
        self.call_history = []

    def set_contact(self, relation: str, phone: str, name: Optional[str] = None):
        """
        Set or update a family contact

        Args:
            relation: "mom" or "dad"
            phone: Phone number
            name: Optional custom name
        """
        if relation.lower() in self.contacts:
            self.contacts[relation.lower()]["phone"] = phone
            if name:
                self.contacts[relation.lower()]["name"] = name
            logger.info(f"Updated contact: {relation} -> {phone}")
        else:
            logger.error(f"Invalid relation: {relation}")

    def call_mom(self) -> Dict[str, any]:
        """
        Call mom (placeholder for actual phone call integration)

        Returns:
            Dict with call status
        """
        return self._initiate_call("mom")

    def call_dad(self) -> Dict[str, any]:
        """
        Call dad (placeholder for actual phone call integration)

        Returns:
            Dict with call status
        """
        return self._initiate_call("dad")

    def _initiate_call(self, relation: str) -> Dict[str, any]:
        """
        Initiate a call to family member

        Args:
            relation: "mom" or "dad"

        Returns:
            Dict with call status and info
        """
        if relation not in self.contacts:
            return {
                "success": False,
                "relation": relation,
                "message": f"Contact not found: {relation}",
            }

        contact = self.contacts[relation]
        phone = contact.get("phone")

        if not phone:
            logger.warning(f"No phone number configured for {relation}")
            return {
                "success": False,
                "relation": relation,
                "name": contact["name"],
                "message": f"No phone number set for {contact['name']}. Please configure in settings.",
            }

        # TODO: Integrate with actual phone system (Twilio, WebRTC, etc.)
        # For now, log the call attempt
        call_time = datetime.now()
        call_record = {
            "relation": relation,
            "name": contact["name"],
            "phone": phone,
            "timestamp": call_time.isoformat(),
            "status": "simulated",  # Change to "connected" when real integration is added
        }

        self.call_history.append(call_record)
        contact["last_called"] = call_time.isoformat()

        logger.info(f"📞 Calling {contact['name']} ({relation}) at {phone}")
        logger.info(
            "NOTE: This is a simulated call. Integrate with Twilio/WebRTC for real calls."
        )

        return {
            "success": True,
            "relation": relation,
            "name": contact["name"],
            "phone": phone,
            "timestamp": call_time.isoformat(),
            "message": f"Calling {contact['name']} now... (Simulated)",
            "note": "Real phone integration pending - see TODO in family_contacts.py",
        }

    def get_contacts(self) -> Dict[str, dict]:
        """Get all family contacts"""
        return self.contacts

    def get_call_history(self, limit: int = 10) -> list:
        """
        Get recent call history

        Args:
            limit: Maximum number of records to return

        Returns:
            List of recent calls
        """
        return self.call_history[-limit:]

    def clear_history(self):
        """Clear call history"""
        self.call_history = []
        for contact in self.contacts.values():
            contact["last_called"] = None
        logger.info("Call history cleared")


# Global instance
_family_contacts = None


def get_family_contacts() -> FamilyContacts:
    """Get or create global family contacts instance"""
    global _family_contacts
    if _family_contacts is None:
        _family_contacts = FamilyContacts()
    return _family_contacts


def reset_family_contacts():
    """Reset family contacts singleton"""
    global _family_contacts
    _family_contacts = None


# Convenience functions
def call_mom() -> Dict[str, any]:
    """Quick function to call mom"""
    return get_family_contacts().call_mom()


def call_dad() -> Dict[str, any]:
    """Quick function to call dad"""
    return get_family_contacts().call_dad()


# Example usage and testing
if __name__ == "__main__":
    print("Family Contacts System - Testing")
    print("=" * 50)

    contacts = get_family_contacts()

    # Set up contacts (in real app, this would come from settings/database)
    contacts.set_contact("mom", "+1-555-0101", "Mom")
    contacts.set_contact("dad", "+1-555-0102", "Dad")

    # Test calling
    print("\nTest 1: Calling Mom")
    result = call_mom()
    print(f"Result: {result}")

    print("\nTest 2: Calling Dad")
    result = call_dad()
    print(f"Result: {result}")

    print("\nCall History:")
    for call in contacts.get_call_history():
        print(f"  - Called {call['name']} at {call['timestamp']}")

    print("\n" + "=" * 50)
    print("NOTE: Phone calls are currently simulated.")
    print("To enable real calling, integrate with:")
    print("  - Twilio API (https://www.twilio.com/docs/voice)")
    print("  - WebRTC for browser-based calls")
    print("  - VoIP service integration")
