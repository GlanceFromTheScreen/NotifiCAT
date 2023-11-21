import dateparser

##############################
# NOW WE HAVE ARRAYS LIKE
# [['catch', 'NTFY'], ['bus', 'NTFY'], ['tomorrow', 'DATE'], ['13:45', 'TIME']]
# AND WE HAVE TO TRANSFORM WORDS AS 'tomorrow' TO RELEVANT DATE
##############################


def get_dict_of_data(arr):
    dict_ = {'NTFY': "", 'DATE': "", 'TIME': ""}
    for record in arr:
        for key in dict_:
            if record[1] == key:
                dict_[key] += record[0] + " "

    dict_['DATE'] = dateparser.parse(dict_['DATE'])
    dict_['TIME'] = dateparser.parse(dict_['TIME'])

    return dict_


