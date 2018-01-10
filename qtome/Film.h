#pragma once

#include <QObject>

class QFilm : public QObject
{
	Q_OBJECT

public:
	QFilm(QObject* pParent = NULL);
	QFilm(const QFilm& Other);
	QFilm& operator=(const QFilm& Other);

	int				GetWidth(void) const;
	void			SetWidth(const int& Width);
	int				GetHeight(void) const;
	void			SetHeight(const int& Height);
	float			GetExposure(void) const;
	void			SetExposure(const float& Exposure);
	int				GetExposureIterations(void) const;
	void			SetExposureIterations(const int& ExposureIterations);
	bool			GetNoiseReduction(void) const;
	void			SetNoiseReduction(const bool& NoiseReduction);
	bool			IsDirty(void) const;
	void			UnDirty(void);

signals:
	void Changed(const QFilm& Film);

private:
	int			m_Width;
	int			m_Height;
	float		m_Exposure;
	int			m_ExposureIterations;
	bool		m_NoiseReduction;
	int			m_Dirty;
};
