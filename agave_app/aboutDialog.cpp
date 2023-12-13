#include "aboutDialog.h"

#include <QApplication>
#include <QDialogButtonBox>
#include <QLabel>
#include <QTextBrowser>
#include <QVBoxLayout>

AboutDialog::AboutDialog()
  : QDialog()
{
  setWindowTitle("About AGAVE");
  auto layout = new QVBoxLayout(this);
  auto label = new QLabel(this);

  label->setText("<h1><b>AGAVE</b></h1>"
                 "<i>Advanced GPU Accelerated Volume Explorer</i>"
                 "<h4><b>Version " +
                 qApp->applicationVersion() + "</b></h4>");
  label->setAlignment(Qt::AlignCenter);
  layout->addWidget(label);

  // label = new QLabel(this);
  // label->setText(
  //   "<a href=\"https://forum.image.sc/tag/agave\">Support forum</a><br/>"
  //   "<a href=\"https://allen-cell-animated.github.io/agave\">Online documentation</a><br/>"
  //   "<a href=\"https://github.com/allen-cell-animated/agave/issues\">Report a bug or make a request</a><br/>"
  //   "<a href=\"https://github.com/allen-cell-animated/agave\">Source code</a><br/>"
  //   "");
  // label->setTextInteractionFlags(Qt::TextBrowserInteraction);
  // label->setOpenExternalLinks(true);
  // layout->addWidget(label);

  QString agaveUrl = "https://github.com/allen-cell-animated/agave";

  auto text = new QLabel(this);
  text->setText(
    "AGAVE is a desktop volume viewer that uses path trace rendering and cinematic lighting techniques to generate "
    "images with a high degree of resolution and clarity. It is designed and optimized for multi-channel OME-TIFF or "
    "OME-Zarr files.<br><br>"
    "<a href=\"https://www.allencell.org/pathtrace-rendering.html\">Visit our website</a> to learn more and download "
    "the latest version.");
  text->setFrameShape(QFrame::Panel);
  text->setFrameShadow(QFrame::Sunken);
  text->setTextInteractionFlags(Qt::TextBrowserInteraction);
  text->setOpenExternalLinks(true);
  text->setWordWrap(true);
  layout->addWidget(text);

  // label = new QLabel(this);
  // label->setText("If you use AGAVE in your research, please cite the following:");
  // layout->addWidget(label);

  // auto citationtext = new QTextBrowser(this);
  // citationtext->setText(
  //   "Daniel Toloudis, AGAVE Contributors (2023). AGAVE: Advanced GPU Accelerated Volume Explorer (Version " +
  //   qApp->applicationVersion() +
  //   ") [Computer software]. Allen Institute for Cell Science. <a "
  //   "href=\"" +
  //   agaveUrl +
  //   "\"></a><br/>"
  //   "<br/>"
  //   "bibtex:<br/>"
  //   "<br/>"
  //   "<tt>@software{agave,<br/>"
  //   "&nbsp;&nbsp;author    = {Toloudis, Daniel and AGAVE Contributors},<br/>"
  //   "&nbsp;&nbsp;title     = {AGAVE: Advanced GPU Accelerated Volume Explorer},<br/>"
  //   "&nbsp;&nbsp;year      = {2023},<br/>"
  //   "&nbsp;&nbsp;version = {" +
  //   qApp->applicationVersion() +
  //   "},<br/>"
  //   "&nbsp;&nbsp;url       = {" +
  //   agaveUrl +
  //   "}<br/>"
  //   "&nbsp;&nbsp;organization = {Allen Institute for Cell Science}<br/>"
  //   "&nbsp;&nbsp;note      = {Computer Software}<br/>"
  //   "}</tt><br/>");
  // citationtext->setFrameShape(QFrame::Panel);
  // citationtext->setFrameShadow(QFrame::Sunken);
  // citationtext->setOpenExternalLinks(true);
  // citationtext->viewport()->setAutoFillBackground(false);
  // layout->addWidget(citationtext);

  label = new QLabel(this);
  label->setText(
    "AGAVE is made possible through the hard work and dedication of engineers, designers, and scientists at the Allen "
    "Institute for Cell Science and through the continued philanthropy of the Paul Allen estate."
    "<br>"
    "<br>"
    "<b>Copyright 2023 The Allen Institute.  All rights reserved.</b><br>"
    "");
  label->setWordWrap(true);
  layout->addWidget(label);

  auto button = new QDialogButtonBox(QDialogButtonBox::Ok, this);
  button->setCenterButtons(true);
  layout->addWidget(button);
  connect(button, &QDialogButtonBox::accepted, this, &QDialog::accept);

  setLayout(layout);
}

AboutDialog::~AboutDialog() {}