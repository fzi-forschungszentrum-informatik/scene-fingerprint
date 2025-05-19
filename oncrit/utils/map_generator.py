# this is for emacs file handling -*- mode: python++; indent-tabs-mode: nil -*-
#
# -- BEGIN LICENSE BLOCK ----------------------------------------------
# -- END LICENSE BLOCK ------------------------------------------------
#
# ---------------------------------------------------------------------
# !\file
#
# \author  Tobias Fleck <tfleck@fzi.de>
# \author  Maximilian Zipfl <zipfl@fzi.de>
#
# ---------------------------------------------------------------------
import xml.etree.ElementTree as ET
import utm
import numpy as np
import matplotlib.pyplot as plt


##
# @brief Simple class structure to read lanelet data from osm files.
class Lanelet:
    def __init__(self):
        ##
        # @brief left waypoints in lat/lon
        self._left = []

        ##
        # @brief left bound of lanelet in metric/Caretsian coordinates w.r.t the
        # reference point
        self._left_metric = []

        ##
        # @brief right waypoitns in latlon
        self._right = []

        ##
        # @brief right bound of lanelet in metric/Cartesian coordinates w.r.t the
        # reference point
        self._right_metric = []

        ##
        # @brief GNSS-Reference point for metric conversion to local utm
        self._gnss_ref = None

        ##
        # @brief reference point in metric UTM coordinates.
        self._gnss_ref_metric = None

        ##
        # @brief id of the lanelet.
        self._id = -1

        ##
        # @brief subtype of the lanelet.
        self._subtype = None

        ##
        # @brief region attribute of the lanelet.
        self._region = None

        ##
        # @brief location attribute of the lanelet.
        self._location = None

    ##
    # @brief
    #
    # @param p
    # @param margin
    # @param bounds = (xmin, xmax, ymin, ymax)
    #
    # @return
    def __point_in_region(self, p, margin, bounds):
        x = p[0]
        y = p[1]
        return x >= bounds[0] - margin \
            and x <= bounds[1] + margin \
            and y >= bounds[2] - margin \
            and y <= bounds[3] + margin

    ##
    # @brief
    #
    # @param margin
    # @param bounds = (xmin, xmax, ymin, ymax)
    #
    # @return
    def is_in_view_region(self, margin, bounds):
        for l in self._left_metric:
            if self.__point_in_region(l, margin, bounds):
                return True

        for l in self._right_metric:
            if self.__point_in_region(l, margin, bounds):
                return True

        return False

    def compute_metric(self):
        if self._gnss_ref is None:
            return False
        else:
            # TODO add handling of different hemispheres and different utm zones
            for lat, lon in self._left:
                metric_x, metric_y, zone, hemisphere = utm.from_latlon(lat, lon)
                self._left_metric.append([metric_x - self._gnss_ref_metric[0],
                                          metric_y - self._gnss_ref_metric[1]])

            for lat, lon in self._right:
                metric_x, metric_y, zone, hemisphere = utm.from_latlon(lat, lon)
                self._right_metric.append([metric_x - self._gnss_ref_metric[0],
                                           metric_y - self._gnss_ref_metric[1]])

            self._left_metric = np.array(self._left_metric)
            self._right_metric = np.array(self._right_metric)

            return True

    @property
    def left(self):
        return self._left

    @property
    def left_metric(self):
        return self._left_metric

    @property
    def subtype(self):
        return self._subtype

    @property
    def region(self):
        return self._region

    @property
    def right(self):
        return self._right

    @property
    def right_metric(self):
        return self._right_metric

    @property
    def location(self):
        return self._location

    @property
    def id(self):
        return self._id

    @property
    def gnss_ref(self):
        return self._gnss_ref

    @left.setter
    def left(self, left):
        self._left = left

    @gnss_ref.setter
    def gnss_ref(self, gnss_ref):
        self._gnss_ref = gnss_ref
        self._gnss_ref_metric = utm.from_latlon(self._gnss_ref[0], self._gnss_ref[1])

    @right.setter
    def right(self, right):
        self._right = right

    @id.setter
    def id(self, id):
        self._id = id

    @subtype.setter
    def subtype(self, subtype):
        self._subtype = subtype

    @region.setter
    def region(self, region):
        self._region = region

    @location.setter
    def location(self, location):
        self._location = location

    def __str__(self):
        str01 = 'id:                  {}\n'.format(self.id)
        str02 = 'region:              {}\n'.format(self.region)
        str03 = 'location:            {}\n'.format(self.location)
        str04 = 'subtype:             {}\n'.format(self.subtype)
        str05 = 'way length (l):      {}\n'.format(len(self.left))
        str06 = 'way length (r):      {}\n'.format(len(self.right))
        return str01 + str02 + str03 + str04 + str05 + str06

    def __repr__(self):
        return self.__str__()

##
# @brief Returns tag of lanelet
#
# @param xml element
#
# @return type of lanelet (string)


def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None

##
# @brief Reads lanelet map information from a given filename.
#
# @param filename
# @param gnss_ref global spherical coordinates of the lanelet map are converted
# to a Cartesian coordinate system that is located in the position gnss_ref and
# has ENU orientation (x = east, y = north).
# @param bounds=(xmin, xmax, ymin, ymax)
#
# @return


def read_lanelets_standalone(filename, bounds, gnss_ref=None):
    tree = ET.parse(filename)
    root = tree.getroot()
    lanelet_nodes = root.findall('relation/tag[@v = "lanelet"]/..')
    lanelets = []

    for ll in lanelet_nodes:
        lanelet = Lanelet()
        lanelet.id = ll.attrib['id']

        location = ll.find('tag[@k = "location"]')
        if not location is None:
            lanelet.location = location.attrib['v']

        region = ll.find('tag[@k = "region"]')
        if not region is None:
            lanelet.region = region.attrib['v']

        subtype = ll.find('tag[@k = "subtype"]')
        if not subtype is None:
            lanelet.subtype = subtype.attrib['v']

        way_left_id = -1
        left_id = ll.find('member[@role = "left"]')
        if left_id != None:
            way_left_id = left_id.attrib['ref']

        way_right_id = -1
        right_id = ll.find('member[@role = "right"]')
        if right_id != None:
            way_right_id = right_id.attrib['ref']

        right_way = root.find('way[@id = "{}"]'.format(way_right_id))
        left_way = root.find('way[@id = "{}"]'.format(way_left_id))
        right_node_refs = right_way.findall('nd')
        left_node_refs = left_way.findall('nd')

        lanelet.right_type = get_type(right_way)
        lanelet.left_type = get_type(right_way)

        right_positions = []
        for n in right_node_refs:
            node = root.find('node[@id = "{}"]'.format(n.attrib['ref']))
            right_positions.append([float(node.attrib['lat']), float(node.attrib['lon'])])

        left_positions = []
        for n in left_node_refs:
            node = root.find('node[@id = "{}"]'.format(n.attrib['ref']))
            left_positions.append([float(node.attrib['lat']), float(node.attrib['lon'])])

        lanelet.right = right_positions
        lanelet.left = left_positions

        if not gnss_ref is None:
            lanelet.gnss_ref = gnss_ref
            lanelet.compute_metric()

        if lanelet.is_in_view_region(margin=10, bounds=bounds):
            lanelets.append(lanelet)
    return lanelets


def lanelet_to_color(l):
    colors = []
    linewidth = []
    for type in (l.left_type, l.right_type):
        if type == "curbstone":
            colors.append("white")
            linewidth.append(1)
        elif type == "line_thin":
            colors.append("white")
            linewidth.append(0.8)
        elif type == "line_thick":
            colors.append("white")
            linewidth.append(1.2)
        elif type == "virtual":
            colors.append("white")
            linewidth.append(0.4)
        else:
            colors.append("white")
            linewidth.append(1)
    return colors[0], linewidth[0], colors[1], linewidth[1]


def draw_bound(bound_data, fig, color, linewidth):
    metric_x = bound_data[:, 0]
    metric_y = bound_data[:, 1]
    plt.plot(metric_x.tolist(), metric_y.tolist(), color, linewidth=linewidth)
    return fig


def draw_lanelets(lanelet_data, fig):
    for l in lanelet_data:
        color_left, linewidth_left, color_right, linewidth_right = lanelet_to_color(l)
        fig = draw_bound(l.left_metric, fig, color_left, linewidth_left)
        fig = draw_bound(l.right_metric, fig, color_right, linewidth_right)
    return fig


def create_figure(lanelets):
    plt.style.use('dark_background')
    plt.axis('off')
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    plt.gca().set_aspect('equal', adjustable='box')
    fig = draw_lanelets(lanelets, fig)
    y_lim = plt.gca().get_ylim()
    x_lim = plt.gca().get_xlim()
    return fig, x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], x_lim[0], y_lim[0]
