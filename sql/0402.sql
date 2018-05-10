-- talking4.mst_dayiphourapp
SELECT
  day, hour, ip, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct hour) uq_hour,
  count(distinct channel) uq_channel,
  count(distinct app) uq_app,
  count(distinct device) uq_device,
FROM `talking2.train_test`
GROUP BY day, hour, ip

-- talking4.mst_dayhouripchannel
SELECT
  day, hour, ip, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel

-- talking4.mst_dayhouripapp
SELECT
  day, hour, ip, app, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct os) uq_os
FROM `talking2.train_test`
GROUP BY day, hour, ip, app

-- talking4.mst_dayhouripos
SELECT
  day, hour, ip, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os

-- talking4.mst_dayhouripdevice
SELECT
  day, hour, ip, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, device

###
-- talking4.mst_dayhouripappchannel
SELECT
  day, hour, ip, app, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, channel

-- talking4.mst_dayhouripchannelos
SELECT
  day, hour, ip, channel, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel, os

-- talking4.mst_dayhouripchanneldevice
SELECT
  day, hour, ip, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, channel, device


-- talking4.mst_dayhouripappos
SELECT
  day, hour, ip, app, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, os

-- talking4.mst_dayhouripappdevice
SELECT
  day, hour, ip, app, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, device

-- talking4.mst_dayhouriosdevice
SELECT
  day, hour, ip, device, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff,
  count(distinct app) as uq_app
FROM `talking2.train_test`
GROUP BY day, hour, ip, device, os
###

-- talking4.mst_nochannel
SELECT
  day, hour, ip, os, app, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, device

-- talking4.mst_noapp
SELECT
  day, hour, ip, os, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, channel, device

-- talking4.mst_noos
SELECT
  day, hour, ip, app, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, app, channel, device

-- talking4.mst_nodevice
SELECT
  day, hour, ip, os, app, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, channel
###

-- talking4.mst_all
SELECT
  day, hour, ip, os, app, channel, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, ip, os, app, channel, device



####

-- talking4.mst_dayhourchannel
SELECT
  day, hour, channel, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, channel

-- talking4.mst_dayhourapp
SELECT
  day, hour, app, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, app

-- talking4.mst_dayhouros
SELECT
  day, hour, os, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, os

-- talking4.mst_dayhourdevice
SELECT
  day, hour, device, count(1) cnt,
  TIMESTAMP_DIFF(MAX(click_time), MIN(click_time), SECOND) as diff
FROM `talking2.train_test`
GROUP BY day, hour, device


####
-- talking4.mst_app
SELECT
  app,
  count(1) as cnt,
  count(distinct channel) uq_channel
FROM `talking2.train_test`
GROUP BY app

####


-- takling4.train_test2
SELECT
  t.click_id,
  t.is_attributed ,
  t.day,
  t.span,
  t.hour,
  t.ip,
  t.os,
  t.app,
  t.channel,
  t.device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time) as row_ip,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os order by click_time) as row_os,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time) as row_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel order by click_time) as row_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device order by click_time) as row_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app order by click_time) as row_os_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.channel order by click_time) as row_os_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.device order by click_time) as row_os_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.channel order by click_time) as row_app_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.device order by click_time) as row_app_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel, t.device order by click_time) as row_channel_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.channel, t.device order by click_time) as row_no_os,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.channel, t.device order by click_time) as row_no_app,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time) as row_no_channel,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.channel order by click_time) as row_no_device,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.channel, t.device order by click_time) as row_all,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour order by click_time desc) as row_ip_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os order by click_time desc) as row_os_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app order by click_time desc) as row_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel order by click_time desc) as row_channel_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.device order by click_time desc) as row_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app order by click_time desc) as row_os_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.channel order by click_time desc) as row_os_channel_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.device order by click_time desc) as row_os_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.channel order by click_time desc) as row_app_channel_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.device order by click_time desc) as row_app_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.channel, t.device order by click_time desc) as row_channel_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.app, t.channel, t.device order by click_time desc) as row_no_os_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.channel, t.device order by click_time desc) as row_no_app_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.device order by click_time desc) as row_no_channel_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.channel order by click_time desc) as row_no_device_r,
  ROW_NUMBER() OVER(partition by t.ip, t.day, t.hour, t.os, t.app, t.channel, t.device order by click_time desc) as row_all_r,
  a.cnt as cnt_dayiphourapp,
  c.cnt as cnt_dayhouripchannel,
  p.cnt as cnt_dayhouripapp,
  o.cnt as cnt_dayhouripos,
  d.cnt as cnt_dayhouripdevice,
  cc.cnt as cnt_dayhourchannel,
  pp.cnt as cnt_dayhourapp,
  oo.cnt as cnt_dayhouros,
  dd.cnt as cnt_dayhourdevice,
  cp.cnt as cnt_dayhouripappchannel,
  co.cnt as cnt_dayhouripchannelos,
  cd.cnt as cnt_dayhouripchanneldevice,
  po.cnt as cnt_dayhouripappos,
  pd.cnt as cnt_dayhouripappdevice,
  od.cnt as cnt_dayhouriosdevice,
  nc.cnt as cnt_nochannel,
  np.cnt as cnt_noapp,
  noo.cnt as cnt_noos,
  nd.cnt as cnt_nodevice,
  g.cnt as cnt_all,
  a.diff as diff_dayiphourapp,
  c.diff as diff_dayhouripchannel,
  p.diff as diff_dayhouripapp,
  o.diff as diff_dayhouripos,
  d.diff as diff_dayhouripdevice,
  cp.diff as diff_dayhouripappchannel,
  co.diff as diff_dayhouripchannelos,
  cd.diff as diff_dayhouripchanneldevice,
  po.diff as diff_dayhouripappos,
  pd.diff as diff_dayhouripappdevice,
  od.diff as diff_dayhouriosdevice,
  nc.diff as diff_nochannel,
  np.diff as diff_noapp,
  noo.diff as diff_noos,
  nd.diff as diff_nodevice,
  g.diff as diff_all
FROM
  `talking2.train_test` as t
LEFT OUTER JOIN `talking4.mst_dayiphourapp` as a
ON a.day = t.day and a.hour = t.hour and a.ip = t.ip
LEFT OUTER JOIN `talking4.mst_dayhouripchannel` as c
ON c.day = t.day and c.hour = t.hour and  c.ip = t.ip and c.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripapp` as p
ON p.day = t.day and p.hour = t.hour and p.ip = t.ip and p.app = t.app
LEFT OUTER JOIN `talking4.mst_dayhouripos` as o
ON o.day = t.day and o.hour = t.hour and o.ip = t.ip and o.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhouripdevice` as d
ON d.day = t.day and d.hour = t.hour and d.ip = t.ip and d.device = t.device
--
LEFT OUTER JOIN `talking4.mst_dayhouripappchannel` as cp
ON cp.day = t.day and cp.hour = t.hour and cp.ip = t.ip and cp.app = t.app and cp.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripchannelos` as co
ON co.day = t.day and co.hour = t.hour and co.ip = t.ip and co.os = t.os and co.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripchanneldevice` as cd
ON cd.day = t.day and cd.hour = t.hour and cd.ip = t.ip and cd.device = t.device and cd.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhouripappos` as po
ON po.day = t.day and po.hour = t.hour and po.ip = t.ip and po.app = t.app and po.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhouripappdevice` as pd
ON pd.day = t.day and pd.hour = t.hour and pd.ip = t.ip and pd.app = t.app and pd.device = t.device
LEFT OUTER JOIN `talking4.mst_dayhouriosdevice` as od
ON od.day = t.day and od.hour = t.hour and od.ip = t.ip and od.os = t.os and od.device = t.device
--
LEFT OUTER JOIN `talking4.mst_nochannel` as nc
ON nc.day = t.day and nc.hour = t.hour and nc.ip = t.ip
and nc.os = t.os and nc.app = t.app and nc.device = t.device
LEFT OUTER JOIN `talking4.mst_noapp` as np
ON np.day = t.day and np.hour = t.hour and np.ip = t.ip
and np.os = t.os and np.channel = t.channel and np.device = t.device
LEFT OUTER JOIN `talking4.mst_noos` as noo
ON noo.day = t.day and noo.hour = t.hour and noo.ip = t.ip
and noo.app = t.app and noo.channel = t.channel and noo.device = t.device
LEFT OUTER JOIN `talking4.mst_nodevice` as nd
ON nd.day = t.day and nd.hour = t.hour and nd.ip = t.ip
and nd.os = t.os and nd.app = t.app and nd.channel = t.channel
--
LEFT OUTER JOIN `talking4.mst_all` as g
ON g.day = t.day and g.hour = t.hour and g.ip = t.ip
and g.os = t.os and g.app = t.app and g.channel = t.channel and g.device = t.device
--
LEFT OUTER JOIN `talking4.mst_dayhourchannel` as cc
ON cc.day = t.day and cc.hour = t.hour and cc.channel = t.channel
LEFT OUTER JOIN `talking4.mst_dayhourapp` as pp
ON pp.day = t.day and pp.hour = t.hour and pp.app = t.app
LEFT OUTER JOIN `talking4.mst_dayhouros` as oo
ON oo.day = t.day and oo.hour = t.hour and oo.os = t.os
LEFT OUTER JOIN `talking4.mst_dayhourdevice` as dd
ON dd.day = t.day and dd.hour = t.hour and dd.device = t.device


SELECT
*
FROM
`talking4.train_test2`
WHERE
  day = 8 AND (click_id is null or click_id >= 0)
