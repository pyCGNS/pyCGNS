#!/usr/bin/python
# Taken from :  http://www.amk.ca/files/python/rng-val
# Thanks to  :  A.M. Kuchling
# -*-Python-*-

import sys, popen2

#import libxml2mod
#import libxsltmod
import libxml2


def main ():
    if len(sys.argv) == 1:
        print 'Usage: %s schema file1 file2 ...' % sys.argv[0]
        sys.exit(1)
    schema_name = sys.argv[1]

    if schema_name.endswith('.rnc'):
        schema = run_trang('rnc', schema_name)
    elif schema_name.endswith('.dtd'):
        schema = run_trang('dtd', schema_name)
    elif schema_name.endswith('.rng') or schema_name.endswith('.xml'):
        input = open(schema_name, 'r')
        schema = input.read()
        input.close()
    else:
        print >>sys.stderr, '%s: unknown extension' % schema_name
        sys.exit(1)

    # Parse the schema
    rngp = libxml2.relaxNGNewMemParserCtxt(schema, len(schema))
    rngs = rngp.relaxNGParse()
    
    for filename in sys.argv[2:]:
        validate(rngs, filename)

def run_trang (input, filename):
    "Run trang as a subprocess to convert a schema to RELAX NG"
    r, w = popen2.popen2('trang -I %s -O rng %s /dev/stdout'
                         % (input, filename))
    w.close()
    schema = ""
    while 1:
        chunk = r.read(1024)
        if chunk == "":
            break
        schema += chunk
    r.close()
    return schema


def validate (rngs, filename):
    reader = libxml2.newTextReaderFilename(filename)
    reader.RelaxNGSetSchema(rngs)

    ret = reader.Read()
    while ret == 1:
        ret = reader.Read()

    sys.stdout.write(filename + ': ')
    if ret != 0:
        print "Error parsing document"

    if reader.IsValid() == 1:
        print "valid"
    else:
        print "invalid"


if __name__ == '__main__':
    main()
    
