package com.chj.dafegrin.domain.bert.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
public class BertResponseDTO {
    private String result;
    private Double posProb;
    @JsonProperty("individual_results")
    private List<BertResult> individualResults;
}
