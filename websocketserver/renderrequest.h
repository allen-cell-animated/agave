#ifndef RENDERREQUEST_H
#define RENDERREQUEST_H

#include <QMatrix4x4>
#include <QWebSocket>
#include "renderparameters.h"

class RenderRequest
{
public:
	RenderRequest(QWebSocket *client, RenderParameters parameters, bool debug = false);

	inline QWebSocket *getClient()
	{
		return client;
	}

	inline RenderParameters getParameters()
	{
		return parameters;
	}

	inline int getDuration()
	{
		return this->estimatedDuration;
	}

	inline bool isDebug()
	{
		return debug;
	}

	inline void setActualDuration(int actualDuration)
	{
		this->actualDuration = actualDuration;
	}

	inline int getActualDuration()
	{
		return actualDuration;
	}

private:
	QWebSocket *client;
	RenderParameters parameters;

	//an estimation of how many miliseconds will this request take to process
	int estimatedDuration;

	//how many nanoseconds did it actually take
	int actualDuration;

	bool debug;

};

#endif // RENDERREQUEST_H
