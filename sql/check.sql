SELECT
count(distinct app) app,
count(distinct device ) device ,
count(distinct os ) os,
count(distinct channel ) channel
FROM
`talking.train` 
