import logging
from typing import ClassVar, Dict, Type

from imgcat import imgcat
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConversationContext():
    """Context for a VLM analysis conversation.
    """

    def add_image(self, image: Image.Image):
        """Adds an image to the converation.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the add_image method.")

    def add_text(self, text: str):
        """Adds a text to the conversation.
        """
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the add_text method.")


class DebuggingTextContext(ConversationContext):
    """Context for a VLM analysis conversation that includes debugging text.
    """

    def add_text(self, text: str):
        """Adds a text to the conversation.
        """
        print(text)

    def add_image(self, image: Image.Image):
        """Adds an image to the conversation.
        """
        imgcat(image)


class RcvOpRequest(BaseModel):
    """
    Base class for all ReasonCV ops (operation requests).
    Subclasses must define a `_name` attribute.
    """
    registry: ClassVar[Dict[str, Type["RcvOpRequest"]]] = {}

    function: str

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses that define a _name attribute."""
        super().__init_subclass__(**kwargs)
        # Retrieve _name directly from the class dict to avoid Pydantic conversion.
        op_type = cls.__dict__.get("_name")
        if op_type is not None:
            RcvOpRequest.registry[op_type] = cls
        else:
            raise ValueError(f"Subclass {cls.__name__} must define a _name attribute.")

    def run(self, image: Image.Image, context: ConversationContext):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement the run method.")


class RcvOpFactory:
    """
    Factory methods for building RcvOpRequest instances.
    """

    @classmethod
    def by_name(cls, name: str) -> RcvOpRequest:
        """Looks up the op type in the registry and returns an instance of the corresponding subclass."""
        return RcvOpRequest.registry[name]

    @classmethod
    def from_doc(cls, doc: dict) -> RcvOpRequest:
        """
        Takes a dict that describes an op and returns an instance of the corresponding subclass.
        Dict needs to include "function" field, which will determine which class,
        and any additional fields that the subclass needs.
        """
        op_type = doc["function"]
        op_cls = cls.by_name(op_type)
        out = op_cls(**doc)
        return out
