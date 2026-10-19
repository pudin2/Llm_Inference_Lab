def siguiente_token(tokenizer, logits):
    """Devuelve (id, texto) del token que el modelo predice tras el prompt.

    logits: tensor [batch, tokens, vocabulario]. Se usa la última posición.
    """

    token_id = logits[0, -1].argmax().item()

    return token_id, tokenizer.decode([token_id])
