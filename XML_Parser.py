import xml.etree.ElementTree as ET
from optparse import OptionParser

import re

PGSL_DLDATA_REQ = 1
PGSL_DLDATA_IND = 2
PGSL_ULDATA_IND = 3
PGSL_STATUS_REQ = 4

SS = "\t"


class PgslParser(object):
    def __init__(self, file, tei, tn, msg):
        self.file = file
        self.tei = tei
        self.tn = tn
        self.msg = msg

        self.MSGS = ["Not Defined", "PGSL-DLDATA-REQ", "PGSL-DLDATA-IND", "PGSL-ULDATA-IND", "PGSL-STATUS-IND",
                     "PGSL-DLDATA-RTTI-IND"]

    def parse(self):
        with open("result1.txt", "w") as output_file:
            self.parse_xml(output_file)

    def parse_xml(self, output_file):
#        tei_flag = False
        rec_time = None
        ip_dec = None
        ip_src = None
        packet_num = 0
        for event, packet in ET.iterparse(self.file):
            if packet.tag == 'packet':
                packet_num += 1
                if packet_num % 1000 == 0:
                    print("Packet number:", packet_num)

                for proto in packet.findall("proto"):
                    if proto.get("name") == 'frame':
                        for field in proto.findall('field'):
                            if field.get('name') == 'frame.time':
                                rec_time = field.get('show').split("China")[0]

                    if proto.get("name") == "ip":
                        m = re.search("(\d+\.\d+\.\d+\.\d+), Dst: (\d+\.\d+\.\d+\.\d+)", proto.get("showname"))
                        if m is not None:
                            ip_src = m.group(1)
                            ip_dec = m.group(2)

                    if proto.get("name") == "bp":
                        msg_tei = int(proto.get("showname").split("TEI:")[1])

                    if proto.get("name") == "pgsl":

                        tn = None
                        msg = None
                        frame_num = None
                        tn_map = None
                        error_cause = None
                        error_value = None
                        for field in proto.findall("field"):
                            if field.get("name") == "pgsl.msgdisc":
                                msg = int(field.get("value"))
                            if field.get("name") == "pgsl.tnres":
                                tn = int(field.get("value"))
                            if field.get("name") == "pgsl.afnd" or field.get("name") == "pgsl.afnu":
                                frame_num = int(field.get("show"))
                            if field.get("show").find("TN Bitmap") != -1:
                                tn_map = int(field.get("value"), 16)

                            if msg == PGSL_STATUS_REQ:
                                if field.get("name") == "pgsl.cause":
                                    error_cause = field.get("value")
                                if field.get("name") == "pgsl.add_info":
                                    error_value = field.get("value")

                        # if self.msg is not None and self.msg != msg:
                        #     continue
                        #
                        # if (msg == PGSL_DLDATA_IND or msg == PGSL_STATUS_REQ) and self.tn is not None and self.tn != tn:
                        #     continue

                        output = None
                        if msg == PGSL_DLDATA_REQ:
                            output = rec_time + SS + ip_src + SS + ip_dec + SS + self.MSGS[int(
                                msg)] + SS + str(msg_tei) + SS + str(tn_map) + SS + str(frame_num) + "\n"

                        elif msg == PGSL_DLDATA_IND:
                            output = rec_time + SS + ip_dec + SS + ip_src + SS + self.MSGS[int(
                                msg)] + SS + str(msg_tei) + SS + str(tn) + SS + str(
                                frame_num) + "\n"

                        elif msg == PGSL_STATUS_REQ:
                            output = rec_time + SS + ip_src + SS + ip_dec + SS + self.MSGS[int(
                                msg)] + SS + str(msg_tei) + SS + str(tn) + SS + str(
                                frame_num) + SS + error_cause + SS + error_value + "\n"

                        if output is not None:
                            output_file.write(output)
                packet.clear()


if __name__ == "__main__":
    usage = "usage: %prog [options] xml_file"
    parser = OptionParser(usage=usage)
    parser.add_option("-e", "--TEI", type="int", dest="TEI", help="filter the TEI")
    parser.add_option("-t", "--TN", type="int", dest="TN", help="filter the time slot")
    parser.add_option("-m", "--MSG", type="int", dest="MSG", help="filter PGSL message")

    (options, agrs) = parser.parse_args()

    tei = options.TEI
    tn = options.TN
    msg = options.MSG

    if len(agrs) == 1:
        xml = agrs[0]
    else:
        parser.print_help()
        exit()

    parser = PgslParser(file=xml, tei=tei, tn=tn, msg=msg)
    parser.parse()
