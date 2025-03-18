def parse_values(lines):
    """
    Helper function that reads variable number of lines from a meta file.
    
    @param lines: the lines of text
    """
    # create the empty list of values starting at this key
    values = []
    
    # iterate until the end of the file or the next key        
    while len(lines) and not lines[0].startswith('#'):
        # skip empty lines
        if not len(lines[0]):
            lines.pop(0)
            continue
        # append the value and continue to the next line
        values.append(lines[0])
        lines.pop(0)
            
    # return the list from the meta file 
    return values



def parse_meta_file(meta_filename):
    """
    Reads a meta file and returns a dictionary of attributes.

    @param meta_filename: the meta file to read from disk
    """
    # create an empty dictionary of attributes
    attributes = {}
    
    # open the meta file and read keys and values from dictionary 
    with open(meta_filename, 'r') as fd:
        # remove all new line characters 
        lines = [line.strip() for line in fd.readlines()]

        # iterate over all lines and grab keys and values 
        while len(lines):
            # must begin with a key
            assert (lines[0].startswith('# '))
            key = lines[0].strip('# ')
            # skip to next line 
            lines = lines[1:]
            # read the values associated with this key
            values = parse_values(lines)
            # values can be empty for flags and empty lists 
            
            attributes[key] = values

    return attributes
