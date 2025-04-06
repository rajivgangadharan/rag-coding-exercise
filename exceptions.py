from qdrant_client.http import exceptions as qdrant_exceptions


class VectorStoreError(Exception):
    """
    Need a custom exception class, I do not want to handle every exception in
    main file one after the other. This is necessary because everything is on
    the cloud
    """

    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

    def __str__(self):
        """
        From where did we get an Exception
        """
        if self.original_exception:
            return f"{super().__str__()}. Original exception: {self.original_exception}"
        else:
            return super().__str__()

    @staticmethod
    def handle_qdrant_exception(e):
        match e:
            case qdrant_exceptions.UnexpectedResponse():
                return VectorStoreError(
                    f"Unexpected Qdrant response (status: {e.status_code}, reason: {e.reason_phrase}): {e.content}",
                    original_exception=e,
                )
            case qdrant_exceptions.ResponseHandlingException():
                return VectorStoreError(
                    f"Error handling Qdrant response: {e.source}", original_exception=e
                )
            case qdrant_exceptions.ApiException():
                return VectorStoreError(f"Qdrant API error: {e}", original_exception=e)
            case _:
                return VectorStoreError(f"Unexpected error: {e}", original_exception=e)
