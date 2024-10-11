from collections import namedtuple

# types
DataColor = namedtuple('DataColor', ['full', 'sample'])
SetupColor = namedtuple('SetupColor', ["incidenceCurveColor",
                                       "dataColor",
                                       "gatesColors",
                                       "errorStructureColor"])

# setup
setup_colors = \
    [
        SetupColor(incidenceCurveColor='maroon',
                   dataColor=DataColor(full='darkred', sample='brown'),
                   gatesColors=('indianred', 'lightcoral'),
                   errorStructureColor='gainsboro'),

        SetupColor(incidenceCurveColor='peru',
                   dataColor=DataColor(full='sienna', sample='saddlebrown'),
                   gatesColors=('sandybrown', 'peachpuff'),
                   errorStructureColor='gray'),

        SetupColor(incidenceCurveColor='darkgreen',
                   dataColor=DataColor(full='darkolivegreen', sample='olivedrab'),
                   gatesColors=('darkseagreen', 'palegreen'),
                   errorStructureColor='darkgray'),

        SetupColor(incidenceCurveColor='indigo',
                   dataColor=DataColor(full='darkslateblue', sample='rebeccapurple'),
                   gatesColors=('darkviolet', 'thistle'),
                   errorStructureColor='dimgray'),

        SetupColor(incidenceCurveColor='navy',
                   dataColor=DataColor(full='darkblue', sample='midnightblue'),
                   gatesColors=('royalblue', 'cornflowerblue'),
                   errorStructureColor='slategray'),

        SetupColor(incidenceCurveColor='darkgoldenrod',
                   dataColor=DataColor(full='goldenrod', sample='gold'),
                   gatesColors=('darkkhaki', 'khaki'),
                   errorStructureColor='silver'),
    ]
