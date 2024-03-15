def member_id2number(member_id):
    '''
    Args:
        member_id (str)
    Returns:
        (int): No. of ensemble members
    '''
    return int(member_id.replace('enstat', ''))