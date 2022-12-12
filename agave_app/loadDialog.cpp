#include "loadDialog.h"

#include <QSpinBox>
#include <QTreeWidget>
#include <QVBoxLayout>
#include <QWidget>

LoadDialog::LoadDialog(std::string path, QWidget* parent)
  : QDialog(parent)
{
  setWindowTitle(tr("Load Settings"));

  mSceneInput = new QSpinBox(this);
  mSceneInput->setMinimum(0);
  mSceneInput->setMaximum(65536);
  mSceneInput->setValue(0);

  struct ZarrMultiscaleDims
  {
    std::vector<float> scale;
    std::vector<int64_t> shape;
    std::string dtype;
    std::string path;
  };
  ZarrMultiscaleDims dims[] = { { { 1, 1, 1 }, { 100, 100, 100 }, "tensorstore::DataType::uint8", "0" },
                                { { 2, 2, 2 }, { 50, 50, 50 }, "tensorstore::DataType::uint8", "1" },
                                { { 4, 4, 4 }, { 25, 25, 25 }, "tensorstore::DataType::uint8", "2" } };
  mMetadataTree = new QTreeWidget(this);
  mMetadataTree->setColumnCount(2);
  mMetadataTree->setHeaderLabels(QStringList() << "Key"
                                               << "Value");
  std::string dimstring = "ZYX";
  for (auto d : dims) {
    QTreeWidgetItem* item = new QTreeWidgetItem(QStringList() << "Scale" << QString::fromStdString(d.path));
    for (size_t i = 0; i < d.shape.size(); ++i) {
      item->addChild(new QTreeWidgetItem(QStringList()
                                         << QString::fromStdString(dimstring.substr(i, 1))
                                         << QString::number(d.shape[i]) + " (" + QString::number(d.scale[i]) + ")"));
    }

    // item->addChild(new QTreeWidgetItem(QStringList()
    //                                    << "Shape"
    //                                    << (QStringList() << QString::number(d.shape[0]) <<
    //                                    QString::number(d.shape[1])
    //                                                      << QString::number(d.shape[2]))
    //                                         .join(",")));
    // item->addChild(new QTreeWidgetItem(QStringList() << "DType" << QString::fromStdString(d.dtype)));
    mMetadataTree->addTopLevelItem(item);
  }

  connect(mSceneInput, SIGNAL(valueChanged(int)), this, SLOT(updateScene(int)));

  QVBoxLayout* layout = new QVBoxLayout(this);

  layout->addWidget(mSceneInput);
  layout->addWidget(mMetadataTree);
  setLayout(layout);
}

void
LoadDialog::updateScene(int value)
{
  // mSettings.mSceneIndex = value;
}
