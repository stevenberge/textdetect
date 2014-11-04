OTHER_FILES += \
    CMakeLists.txt \
    utils.h.gch \
    text_extraction \
    textdetect.pro.user \
    test \
    RegionLine.hpp.gch \
    Makefile \
    cmake_install.cmake \
    clear.sh \
    bake.sh

SOURCES += \
    template.cpp \
    region_classifier.cpp \
    region.cpp \
    nfa.cpp \
    mser.cpp \
    min_bounding_box.cpp \
    max_meaningful_clustering.cpp \
    group_classifier.cpp \
    fast_clustering.cpp \
    afterline.cpp\
    extend.cpp


HEADERS += \
    utils.h \
    test.pch \
    RegionLine.hpp \
    region_classifier.h \
    region.h \
    mser.h \
    min_bounding_box.h \
    max_meaningful_clustering.h \
    group_classifier.h\
    extend.h

