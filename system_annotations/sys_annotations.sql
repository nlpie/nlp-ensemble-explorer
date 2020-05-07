use mhealth;

SELECT u.begin, u.end, null as concept, u.cui, u.confidence as score, u.tui as semType, u.note_id, u.type, u.system, -1 as polarity FROM bio_biomedicus_UmlsConcept u left join bio_biomedicus_Negated n
on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id
where n.begin is not null and n.end is not null and n.note_id is not null
union distinct
SELECT u.begin, u.end, null as concept, u.cui, u.confidence as score, u.tui as semType, u.note_id, u.type, u.system, 1 as polarity FROM bio_biomedicus_UmlsConcept u left join bio_biomedicus_Negated n
on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id
where n.begin is null and n.end is null and n.note_id is null

union

select u.begin, u.end, u.concept, u.cui, u.concept_prob as score, u.semanticTag as semType, u.note_id, u.type, u.system, -1 as polarity from cla_edu_ClampNameEntityUIMA u left join cla_edu_ClampRelationUIMA r
on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id
where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null
union distinct
select u.begin, u.end, u.concept, u.cui, u.concept_prob as score, u.semanticTag as semType, u.note_id, u.type, u.system, 1 as polarity from cla_edu_ClampNameEntityUIMA u left join cla_edu_ClampRelationUIMA r
on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id
where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null

union

select c.begin, c.end, c.preferred as concept, c.cui, abs(c.score) as score, c.semanticTypes as semType, c.note_id, c.type, c.system, -1 as polarity from met_org_candidate c left join met_org_Negation n
on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id
where n.begin is not null and n.end is not null and n.note_id is not null
union distinct
select c.begin, c.end, c.preferred as concept, c.cui, abs(c.score) as score, c.semanticTypes as semType, c.note_id, c.type, c.system, 1 as polarity from met_org_candidate c left join met_org_Negation n
on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id
where n.begin is null and n.end is null and n.note_id is null

union

select  begin, end, concept, cui, null as score, substring_index(substring_index(type,',', 7),'.', -(1)) as semtype,
		note_id, 'ctakes_mentions' as type,  `system`, polarity  
    from cta_org_anatomicalsitemention 
union distinct 
select begin, end, concept, cui, null as score, substring_index(substring_index(type,',', 7),'.', -(1)) as semtype,
	   note_id, 'ctakes_mentions' as type,  `system`, polarity 
    from cta_org_diseasedisordermention 
union distinct 
select begin, end, concept, cui, null as score, substring_index(substring_index(type,',', 7),'.', -(1)) as semtype,
	   note_id, 'ctakes_mentions' as type,  `system`, polarity 
    from cta_org_medicationmention 
union distinct 
select begin, end, concept, cui, null as score, substring_index(substring_index(type,',', 7),'.', -(1)) as semtype,
	   note_id, 'ctakes_mentions' as type,  `system`, polarity 
    from cta_org_proceduremention 
union distinct 
select begin, end, concept, cui, null as score, substring_index(substring_index(type,',', 7),'.', -(1)) as semtype,
	   note_id, 'ctakes_mentions' as type,  `system`, polarity 
    from cta_org_signsymptommention;
    
    SELECT begin, end, term as concept, cui, similarity as score, semtypes as semType, note_id, type, `system`, 0 as polarity FROM mhealth.qumls_all;




