def escape_curly_braces(text):
    return text.replace('{', '{{').replace('}', '}}')

def escape_messages_curly_braces(messages: list[tuple[str,str]]) -> list[tuple[str,str]]:
    return [(escape_curly_braces(author), escape_curly_braces(message)) for author, message in messages]