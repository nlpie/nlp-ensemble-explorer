SELECT u.begin, u.end, u.cui, u.confidence as score, u.tui as semType, u.note_id, u.type, u.system, -1 as polarity FROM mhealth.bio_biomedicus_UmlsConcept u left join mhealth.bio_biomedicus_Negated n
on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id
where n.begin is not null and n.end is not null and n.note_id is not null
union distinct
SELECT u.begin, u.end, u.cui, u.confidence as score, u.tui as semType, u.note_id, u.type, u.system, 1 as polarity FROM mhealth.bio_biomedicus_UmlsConcept u left join mhealth.bio_biomedicus_Negated n
on u.begin = n.begin and u.end = n.end and u.note_id = n.note_id
where n.begin is null and n.end is null and n.note_id is null;


select u.begin, u.concept, u.cui, u.end, u.note_id, u.semanticTag, u.system, u.type, u.concept_prob as score, -1 as polarity from mhealth.cla_edu_ClampNameEntityUIMA u left join mhealth.cla_edu_ClampRelationUIMA r
on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id
where u.assertion = 'absent' and r.begin is not null and r.end is not null and r.note_id is not null
union distinct
select u.begin, u.concept, u.cui, u.end, u.note_id, u.semanticTag, u.system, u.type, u.concept_prob as score, 1 as polarity from mhealth.cla_edu_ClampNameEntityUIMA u left join mhealth.cla_edu_ClampRelationUIMA r
on u.begin = r.begin and u.end = r.end and u.note_id = r.note_id
where (u.assertion = 'present' or u.assertion is null) and r.begin is null and r.end is null and r.note_id is null;


select c.begin, c.preferred as concept, c.cui, c.end, c.note_id, abs(c.score) as score, c.system, c.type, -1 as polarity, c.cui, c.semanticTypes from mhealth.met_org_candidate c left join mhealth.met_org_Negation n
on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id
where n.begin is not null and n.end is not null and n.note_id is not null
union distinct
select c.begin, c.preferred as concept, c.cui, c.end, c.note_id, abs(c.score) as score, c.system, c.type, 1 as polarity, c.cui, c.semanticTypes from mhealth.met_org_candidate c left join mhealth.met_org_Negation n
on c.begin = n.begin and c.end = n.end and c.note_id = n.note_id
where n.begin is null and n.end is null and n.note_id is null;

select  begin, end, 'ctakes_mentions' as type, cui, `system`, note_id, polarity, concept,
        substring_index(substring_index(type,',', 7),'.', -(1)) as semtypes
    from mhealth.cta_org_anatomicalsitemention 
union distinct 
select begin, end, 'ctakes_mentions' as type, cui, `system`, note_id, polarity, concept,
        substring_index(substring_index(type,',', 7),'.', -(1)) as semtypes
    from mhealth.cta_org_diseasedisordermention 
union distinct 
select begin, end, 'ctakes_mentions' as type, cui, `system`, note_id, polarity, concept,
        substring_index(substring_index(type,',', 7),'.', -(1)) as semtypes
    from mhealth.cta_org_medicationmention 
union distinct 
select begin, end, 'ctakes_mentions' as type, cui, `system`, note_id, polarity, concept,
        substring_index(substring_index(type,',', 7),'.', -(1)) as semtypes
    from mhealth.cta_org_proceduremention 
union distinct 
select begin, end, 'ctakes_mentions' as type, cui, `system`, note_id, polarity, concept,
        substring_index(substring_index(type,',', 7),'.', -(1)) as semtypes
    from mhealth.cta_org_signsymptommention




