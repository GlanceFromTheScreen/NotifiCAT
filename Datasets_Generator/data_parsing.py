import dateparser

##############################
# NOW WE HAVE ARRAYS LIKE
# [['catch', 'NTFY'], ['bus', 'NTFY'], ['tomorrow', 'DATE'], ['13:45', 'TIME']]
# AND WE HAVE TO TRANSFORM WORDS AS 'tomorrow' TO RELEVANT DATE
##############################


def get_dict_of_data(arr):
    dict_ = {'NTFY': "", 'DATE': "", 'TIME': ""}
    for record in arr:
        if record[1] == "NTFY" or record[1] == "GPE":
            dict_["NTFY"] += record[0] + " "
        elif record[1] == "DATE":
            dict_["DATE"] += record[0] + " "
        elif record[1] == "TIME":
            dict_["TIME"] += record[0] + " "

    dict_['DATE'] = dateparser.parse(dict_['DATE'], settings={'PREFER_DATES_FROM': 'future'})
    dict_['TIME'] = dateparser.parse(dict_['TIME'], settings={'PREFER_DATES_FROM': 'future'})

    return dict_


