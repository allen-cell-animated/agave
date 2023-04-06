#pragma once

#include "Controls.h"
#include "Section.h"

#include "renderlib/FileReader.h"
#include "renderlib/VolumeDimensions.h"

#include <QComboBox>
#include <QDialog>
#include <QEvent>
#include <QLineEdit>
#include <QListView>
#include <QStandardItemModel>
#include <QStyledItemDelegate>

class QIntSlider;
class QLabel;
class QListWidget;
class QSpinBox;
class QTreeWidget;

class IFileReader;
class RangeWidget;
class Section;

class LoadDialog : public QDialog
{
  Q_OBJECT

public:
  LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, uint32_t scene, QWidget* parent = Q_NULLPTR);
  ~LoadDialog();

  LoadSpec getLoadSpec() const;
  int getMultiscaleLevelIndex() const { return mSelectedLevel; }

  QSize sizeHint() const override { return QSize(400, 400); }
private slots:
  void updateScene(int value);
  void updateMultiresolutionLevel(int level);
  void updateChannels();

  void accept() override;

private:
  std::string mPath;
  int mScene;
  int mTime;
  std::vector<MultiscaleDims> mDims;
  int mSelectedLevel;

  QSpinBox* mSceneInput;
  QComboBox* mMultiresolutionInput;
  QIntSlider* m_TimeSlider;
  QListWidget* mChannels;
  Section* mChannelsSection;
  QTreeWidget* mMetadataTree;
  QLabel* mVolumeLabel;
  QLabel* mMemoryEstimateLabel;
  RangeWidget* m_roiX;
  RangeWidget* m_roiY;
  RangeWidget* m_roiZ;
  Section* m_roiSection;
  // show multiresolutions

  // select region of interest in zyx
  // select any set of channels
  // start with a single timepoint

  void updateMemoryEstimate();
  void updateMultiresolutionInput();
  std::vector<uint32_t> getCheckedChannels() const;
  void populateChannels(int level);

  IFileReader* m_reader;
};
