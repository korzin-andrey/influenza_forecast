from pylatex import Document, PageStyle, Head, MiniPage, Foot, LargeText, \
    MediumText, LineBreak, Command, Package, simple_page_number
from pylatex.utils import bold


# \usepackage[T1]{fontenc}
# \usepackage[utf8]{inputenc}
# \usepackage[russian]{babel}
# \usepackage{lmodern}

def set_latex_packages():
    geometry_options = {"margin": "0.7in"}
    doc = Document(geometry_options=geometry_options)
    packages = [Package('babel', options=['russian']),
                Package('inputenc', options='utf8')]
    # doc.packages.append(Package('fontenc', options=['T1']))
    doc.packages.append(Package('inputenc', options=['utf8']))
    doc.packages.append(Package('babel', options=['russian']))
    doc.packages.append(Package('lmodern'))
    return doc


def generate_header(doc):
    # Add document header
    header = PageStyle("header")
    # Create left header
    with header.create(Head("L")):
        header.append("Дата:")
    # Create center header
    with header.create(Head("C")):
        header.append("ФГБУ НИИ гриппа имени А.А. Смородинцева")

    doc.preamble.append(header)
    doc.change_document_style("header")

    # Add Heading
    with doc.create(MiniPage(align='c')):
        doc.append(
            LargeText(bold("ЕЖЕНЕДЕЛЬНЫЙ НАЦИОНАЛЬНЫЙ БЮЛЛЕТЕНЬ ПО ГРИППУ И ОРВИ")))
        doc.append(LineBreak())
        doc.append(MediumText(
            bold("за 1 неделю 2020 года (30.12.19 -- 05.01.20)")))
        doc.append(LineBreak())

    return doc


def generate_description_multi_age(doc):
    desctription_file = open(
        "description_age_group.tex", encoding='utf-8').read()
    doc.append(desctription_file)
    return doc


if __name__ == '__main__':
    doc = set_latex_packages()
    doc = generate_header(doc)
    doc = generate_description_multi_age(doc)
    doc.generate_pdf("header", clean_tex=False, compiler='pdfLaTeX')
