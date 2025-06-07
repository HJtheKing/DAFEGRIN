package com.chj.dafegrin.entity;

import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Getter
@Setter
public class News {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "news_id")
    private Long id;

    @ManyToOne
    @JsonIgnore
    @JoinColumn(name = "dailyindex_id")
    private DailyIndex dailyIndex;

    @Column(name = "news_date")
    private LocalDateTime date;

    @Column(name = "is_pos")
    private boolean isPos;

    private String content;

    @Column(name = "pos_prob")
    private double posProb;
}
