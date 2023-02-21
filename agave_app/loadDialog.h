#pragma once

#include "renderlib/FileReader.h"
#include "renderlib/VolumeDimensions.h"

#include <QDialog>

class QComboBox;
class QIntSlider;
class QLabel;
class QSpinBox;
class QTreeWidget;

class IFileReader;
class RangeWidget;

class LoadDialog : public QDialog
{
  Q_OBJECT

public:
  LoadDialog(std::string path, const std::vector<MultiscaleDims>& dims, QWidget* parent = Q_NULLPTR);
  ~LoadDialog();

  LoadSpec getLoadSpec() const;
  int getMultiscaleLevelIndex() const { return mSelectedLevel; }

private slots:
  void updateScene(int value);
  void onItemSelectionChanged();
  void updateMultiresolutionLevel(int level);

private:
  std::string mPath;
  int mScene;
  int mTime;
  std::vector<MultiscaleDims> mDims;
  int mSelectedLevel;

  QSpinBox* mSceneInput;
  QComboBox* mMultiresolutionInput;
  QIntSlider* m_TimeSlider;
  QTreeWidget* mMetadataTree;
  QLabel* mMemoryEstimateLabel;
  RangeWidget* m_roiX;
  RangeWidget* m_roiY;
  RangeWidget* m_roiZ;
  // show multiresolutions

  // select region of interest in zyx
  // select any set of channels
  // start with a single timepoint

  void updateMemoryEstimate();
  void updateMultiresolutionInput();

  IFileReader* m_reader;
};
